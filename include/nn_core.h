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
int nn_rank_action_scores(const nn_real *scores, size_t count, size_t *ranked_indices);

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
