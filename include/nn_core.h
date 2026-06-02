#ifndef NN_CORE_H
#define NN_CORE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifndef NN_REAL
#define NN_REAL float
#ifndef NN_REAL_IS_FLOAT
#define NN_REAL_IS_FLOAT 1
#endif
#endif

#ifndef NN_REAL_IS_FLOAT
#define NN_REAL_IS_FLOAT 0
#endif

typedef NN_REAL nn_real;
typedef struct NeuralNet NeuralNet;

#define NN_TRACE_SCHEMA "crlp.trace.v1"
#define NN_TRACE_VERSION 1u
#define NN_TRACE_ACTION_ID_CONTRACT "stable_unique_action_ids"
#define NN_RANKING_POLICY "score_desc_action_id_tiebreak_epsilon"
#define NN_SCORER_ENVELOPE_FEATURE_COUNT 96u
#define NN_SCORER_ENVELOPE_FEATURE_SET "task_role_tagged_action_rule_rows_v1"
#if NN_REAL_IS_FLOAT
#define NN_RANK_SCORE_EPSILON ((nn_real)1e-6f)
#else
#define NN_RANK_SCORE_EPSILON ((nn_real)1e-12)
#endif

typedef enum {
    NN_ACTIVATION_SIGMOID = 0,
    NN_ACTIVATION_RELU = 1,
    NN_ACTIVATION_TANH = 2,
    NN_ACTIVATION_LINEAR = 3,
} NNActivation;

typedef enum {
    NN_LOSS_MSE = 0,
    NN_LOSS_SOFTMAX_CROSS_ENTROPY = 1,
} NNLoss;

typedef struct {
    const size_t *layer_sizes;
    size_t layer_count;
    unsigned int seed;
    NNActivation hidden_activation;
    NNActivation output_activation;
    NNLoss loss;
} NNConfig;

typedef struct {
    uint32_t action_id;
    const nn_real *features;
    nn_real score;
    nn_real probability;
} NNActionCandidate;

typedef enum {
    NN_SCORER_ROLE_ACTION = 0,
    NN_SCORER_ROLE_RULE = 1,
} NNScorerRole;

typedef enum {
    NN_SCORER_DOMAIN_PRIME_GAP_ACTION = 0,
    NN_SCORER_DOMAIN_GRID_PATH_ACTION = 1,
    NN_SCORER_DOMAIN_DELAYED_CLAIM_ACTION = 2,
    NN_SCORER_DOMAIN_ARC_AGI_ACTION = 3,
    NN_SCORER_DOMAIN_PRIME_RULE_PROPOSE = 4,
    NN_SCORER_DOMAIN_GO_POSITION_ACTION = 5,
} NNScorerDomain;

typedef enum {
    NN_SCORER_ENVELOPE_F_BIAS = 0,
    NN_SCORER_ENVELOPE_F_ROLE_RULE = 1,
    NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_RULE = 2,
    NN_SCORER_ENVELOPE_F_ROLE_ACTION = 3,
    NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_ACTION = 4,
    NN_SCORER_ENVELOPE_F_CANDIDATE_INDEX = 5,
    NN_SCORER_ENVELOPE_F_CANDIDATE_COUNT = 6,
    NN_SCORER_ENVELOPE_F_DOMAIN_GRID_ACTION = 7,
    NN_SCORER_ENVELOPE_F_DOMAIN_CLAIM_ACTION = 8,
    NN_SCORER_ENVELOPE_F_DOMAIN_ARC_ACTION = 9,
    NN_SCORER_ENVELOPE_F_DOMAIN_GO_ACTION = 31,
    NN_SCORER_ENVELOPE_F_ACTION_COMMON = 10,
    NN_SCORER_ENVELOPE_F_ACTION_STATE = 20,
    NN_SCORER_ENVELOPE_F_ROLE_PAYLOAD = 32,
    NN_SCORER_ENVELOPE_F_RESIDUAL_FEATURES = NN_SCORER_ENVELOPE_F_ROLE_PAYLOAD,
} NNScorerEnvelopeSlot;

typedef struct {
    NNScorerRole role;
    NNScorerDomain domain;
    size_t candidate_index;
    size_t candidate_count;
    size_t candidate_count_scale;
    int include_role_tag;
    int include_domain_tag;
} NNScorerEnvelopeRow;

typedef struct {
    size_t candidate_count;
    size_t feature_count;
    NNActionCandidate *candidates;
} NNActionSet;

typedef struct {
    uint32_t checkpoint_version;
    uint32_t feature_encoder_version;
    char parent_model_hash[65];
    char training_trace_root[65];
    char benchmark_report_hash[65];
    char notes[256];
    char feature_set[32];
} NNCheckpointMeta;

typedef struct {
    const char *task_name;
    const char *model_hash_hex;
    uint64_t state_hash;
    uint32_t feature_encoder_version;
    size_t candidate_count;
    const uint32_t *action_ids;
    const nn_real *scores;
    const nn_real *probabilities;
    size_t selected_index;
    size_t target_index;
    const size_t *target_indices;
    size_t target_count;
    nn_real loss;
    const char *verifier_result;
    const char *verifier_cost_json;
    uint64_t seed;
    const char *extra_json;
} NNTraceRecord;

typedef struct {
    char schema[32];
    uint32_t trace_version;
    char action_id_contract[32];
    char task_name[64];
    char model_hash_hex[65];
    uint64_t state_hash;
    uint32_t feature_encoder_version;
    size_t candidate_count;
    size_t selected_index;
    size_t target_index;
    uint32_t selected_action_id;
    uint32_t target_action_id;
    nn_real selected_score_margin;
    nn_real rank_score_epsilon;
    char ranking_policy[64];
    nn_real loss;
    char verifier_result[32];
    uint64_t seed;
} NNParsedTraceRecord;

NeuralNet *nn_create(const size_t *layer_sizes, size_t layer_count, unsigned int seed);
NeuralNet *nn_create_config(const NNConfig *config);
NeuralNet *nn_clone(const NeuralNet *net);
void nn_free(NeuralNet *net);

size_t nn_layer_count(const NeuralNet *net);
size_t nn_layer_size(const NeuralNet *net, size_t layer_index);
NNActivation nn_hidden_activation(const NeuralNet *net);
NNActivation nn_output_activation(const NeuralNet *net);
NNLoss nn_loss(const NeuralNet *net);
size_t nn_input_size(const NeuralNet *net);
size_t nn_output_size(const NeuralNet *net);
size_t nn_parameter_count(const NeuralNet *net);
size_t nn_forward_flop_count(const NeuralNet *net);

void nn_forward(NeuralNet *net, const nn_real *input, nn_real *output);
void nn_forward_batch(NeuralNet *net,
                      const nn_real *inputs,
                      nn_real *outputs,
                      size_t sample_count);
void nn_softmax(const nn_real *scores, nn_real *probabilities, size_t count);
nn_real nn_softmax_cross_entropy(const nn_real *scores, const nn_real *targets, size_t count);

int nn_score_action_rows(NeuralNet *net,
                         const nn_real *candidate_inputs,
                         size_t candidate_count,
                         nn_real *scores,
                         nn_real *probabilities);
const char *nn_scorer_envelope_feature_set(void);
const char *nn_scorer_envelope_slot_name(size_t slot);
int nn_scorer_envelope_domain_slot(NNScorerDomain domain, size_t *out_slot);
int nn_scorer_envelope_fill_base(nn_real *row,
                                 size_t row_capacity,
                                 const NNScorerEnvelopeRow *spec);
int nn_rank_action_scores(const nn_real *scores, size_t count, size_t *ranked_indices);
int nn_rank_action_scores_with_ids(const nn_real *scores,
                                   const uint32_t *action_ids,
                                   size_t count,
                                   size_t *ranked_indices,
                                   nn_real score_epsilon);
nn_real nn_action_score_margin(const nn_real *scores, size_t count, size_t selected_index);

void nn_checkpoint_meta_init(NNCheckpointMeta *meta, uint32_t feature_encoder_version);
int nn_checkpoint_meta_set_parent_model_hash(NNCheckpointMeta *meta, const char *value);
int nn_checkpoint_meta_set_training_trace_root(NNCheckpointMeta *meta, const char *value);
int nn_checkpoint_meta_set_benchmark_report_hash(NNCheckpointMeta *meta, const char *value);
int nn_checkpoint_meta_set_notes(NNCheckpointMeta *meta, const char *value);
int nn_checkpoint_meta_set_feature_set(NNCheckpointMeta *meta, const char *value);
int nn_save_checkpoint(const NeuralNet *net,
                       const char *path,
                       const NNCheckpointMeta *meta);
NeuralNet *nn_load_checkpoint(const char *path, NNCheckpointMeta *out_meta);
int nn_hash_checkpoint_file(const char *path, uint8_t out_hash[32]);
int nn_hash_checkpoint_hex(const char *path, char out_hex[65]);

int nn_trace_write_jsonl(FILE *fp, const NNTraceRecord *record);
int nn_trace_parse_jsonl(const char *line,
                         NNParsedTraceRecord *out_record,
                         uint32_t *action_ids,
                         nn_real *scores,
                         nn_real *probabilities,
                         size_t candidate_capacity);
int nn_trace_json_get_u64(const char *line, const char *key, uint64_t *out_value);
int nn_trace_json_get_real(const char *line, const char *key, nn_real *out_value);
int nn_trace_json_get_string(const char *line,
                             const char *key,
                             char *out_value,
                             size_t out_value_size);

#ifdef NN_ENABLE_TEST_API
nn_real nn_test_get_parameter(const NeuralNet *net, size_t index);
int nn_test_set_parameter(NeuralNet *net, size_t index, nn_real value);
int nn_test_has_gradient_buffers(const NeuralNet *net);
#endif

#endif
