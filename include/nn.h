#ifndef NN_H
#define NN_H

#include <stddef.h>

#ifndef NN_REAL
#define NN_REAL float
#ifndef NN_REAL_IS_FLOAT
#define NN_REAL_IS_FLOAT 1
#endif
#endif

#ifndef NN_REAL_IS_FLOAT
#define NN_REAL_IS_FLOAT 0
#endif

#include <stdio.h>
#include <stdint.h>

typedef NN_REAL nn_real;
typedef struct NeuralNet NeuralNet;

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

typedef enum {
    NN_CHECKPOINT_SCALAR_FLOAT32 = 1,
    NN_CHECKPOINT_SCALAR_FLOAT64 = 2,
} NNCheckpointScalarType;

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
    uint32_t feature_encoder_version;
    uint32_t normalization_version;
    uint8_t normalization_hash[32];
    uint8_t parent_model_hash[32];
    uint8_t training_trace_root[32];
    uint8_t training_view_root[32];
    uint8_t benchmark_report_hash[32];
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
} NNTraceRecord;

NeuralNet *nn_create(const size_t *layer_sizes, size_t layer_count, unsigned int seed);
NeuralNet *nn_create_config(const NNConfig *config);
void nn_free(NeuralNet *net);

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
int nn_score_actions(NeuralNet *net,
                     NNActionCandidate *candidates,
                     size_t candidate_count);
int nn_score_action_rows(NeuralNet *net,
                         const nn_real *candidate_inputs,
                         size_t candidate_count,
                         nn_real *scores,
                         nn_real *probabilities);
int nn_rank_actions(const NNActionCandidate *candidates,
                    size_t candidate_count,
                    size_t *ranked_indices);
int nn_save_checkpoint(const NeuralNet *net,
                       const char *path,
                       const NNCheckpointMeta *meta);
NeuralNet *nn_load_checkpoint(const char *path, NNCheckpointMeta *out_meta);
int nn_hash_checkpoint_file(const char *path, uint8_t out_hash[32]);
int nn_trace_write_jsonl(FILE *file, const NNTraceRecord *record);
nn_real nn_train_sample(NeuralNet *net,
                        const nn_real *input,
                        const nn_real *target,
                        nn_real learning_rate);
nn_real nn_train_batch(NeuralNet *net,
                       const nn_real *inputs,
                       const nn_real *targets,
                       size_t batch_size,
                       nn_real learning_rate);
nn_real nn_train(NeuralNet *net,
                 const nn_real *inputs,
                 const nn_real *targets,
                 size_t sample_count,
                 nn_real learning_rate,
                 size_t epochs,
                 int shuffle);
nn_real nn_train_batched(NeuralNet *net,
                         const nn_real *inputs,
                         const nn_real *targets,
                         size_t sample_count,
                         size_t batch_size,
                         nn_real learning_rate,
                         size_t epochs,
                         int shuffle);
nn_real nn_train_softmax_scorer(NeuralNet *net,
                                const nn_real *candidate_inputs,
                                size_t candidate_count,
                                size_t target_index,
                                nn_real learning_rate);
nn_real nn_train_action_scorer(NeuralNet *net,
                               const nn_real *candidate_inputs,
                               size_t candidate_count,
                               size_t target_index,
                               nn_real learning_rate);
nn_real nn_train_softmax_scorer_entropy(NeuralNet *net,
                                        const nn_real *candidate_inputs,
                                        size_t candidate_count,
                                        size_t target_index,
                                        nn_real entropy_weight,
                                        nn_real learning_rate);
nn_real nn_train_softmax_scorer_targets(NeuralNet *net,
                                        const nn_real *candidate_inputs,
                                        size_t candidate_count,
                                        const nn_real *target_distribution,
                                        nn_real learning_rate);
nn_real nn_train_softmax_scorer_weighted(NeuralNet *net,
                                         const nn_real *candidate_inputs,
                                         size_t candidate_count,
                                         size_t selected_index,
                                         nn_real weight,
                                         nn_real learning_rate);
nn_real nn_train_softmax_scorer_cached(NeuralNet *net,
                                       size_t candidate_count,
                                       size_t selected_index,
                                       nn_real weight,
                                       nn_real learning_rate);

#ifdef NN_ENABLE_TEST_API
nn_real nn_test_get_parameter(const NeuralNet *net, size_t index);
int nn_test_set_parameter(NeuralNet *net, size_t index, nn_real value);
int nn_test_has_gradient_buffers(const NeuralNet *net);
#endif

#endif
