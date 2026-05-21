#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include "nn_core.h"

#define NN_NUMERIC_STEP_CERTIFICATE_SCHEMA "crlp.numeric_step_certificate.v1"
#define NN_NEURON_CONTRACT_SCHEMA "crlp.neuron_contract.v1"

typedef enum {
    NN_NUMERIC_STEP_REJECT_NONE = 0,
    NN_NUMERIC_STEP_REJECT_INVALID_ARGUMENT = 1,
    NN_NUMERIC_STEP_REJECT_NONFINITE = 2,
    NN_NUMERIC_STEP_REJECT_PARAMETER_BOUND = 3,
    NN_NUMERIC_STEP_REJECT_PARAMETER_DELTA_BOUND = 4,
    NN_NUMERIC_STEP_REJECT_LOGIT_BOUND = 5,
    NN_NUMERIC_STEP_REJECT_LOGIT_DELTA_BOUND = 6,
    NN_NUMERIC_STEP_REJECT_MARGIN_BOUND = 7,
    NN_NUMERIC_STEP_REJECT_ALLOC_FAILED = 8,
} NNNumericStepRejectReason;

typedef struct {
    int require_finite;
    size_t max_shrink_steps;
    nn_real shrink_factor;
    nn_real max_abs_parameter;
    nn_real max_abs_parameter_delta;
    nn_real max_abs_logit;
    nn_real max_abs_logit_delta;
    nn_real min_selected_margin;
} NNNumericGuardConfig;

typedef struct {
    char schema[48];
    char step_kind[32];
    uint32_t report_version;
    int step_committed;
    NNNumericStepRejectReason rejection_reason;
    size_t shrink_steps;
    size_t parameter_count;
    size_t candidate_count;
    size_t selected_index;
    nn_real attempted_learning_rate;
    nn_real applied_learning_rate;
    nn_real optimizer_loss;
    nn_real loss_before;
    nn_real loss_after;
    nn_real max_abs_parameter_before;
    nn_real max_abs_parameter_after;
    nn_real max_abs_parameter_delta;
    nn_real max_abs_logit_before;
    nn_real max_abs_logit_after;
    nn_real max_abs_logit_delta;
    nn_real selected_margin_before;
    nn_real selected_margin_after;
    int finite_before;
    int finite_after;
} NNTrainingStepCertificate;

typedef struct {
    char schema[32];
    uint32_t report_version;
    size_t layer_index;
    size_t neuron_index;
    size_t input_width;
    nn_real input_min;
    nn_real input_max;
    nn_real pre_activation_min;
    nn_real pre_activation_max;
    nn_real output_min;
    nn_real output_max;
    nn_real weight_l1;
    nn_real derivative_upper_bound;
    nn_real lipschitz_upper_bound;
    int certified;
} NNNeuronContract;

void nn_numeric_guard_config_init(NNNumericGuardConfig *config);
void nn_training_step_certificate_init(NNTrainingStepCertificate *certificate);
const char *nn_numeric_step_reject_reason_name(NNNumericStepRejectReason reason);
int nn_training_step_certificate_write_jsonl(FILE *fp,
                                             const NNTrainingStepCertificate *certificate);

int nn_neuron_contract_interval(const NeuralNet *net,
                                size_t layer_index,
                                size_t neuron_index,
                                nn_real input_min,
                                nn_real input_max,
                                NNNeuronContract *out_contract);
int nn_neuron_contract_write_jsonl(FILE *fp, const NNNeuronContract *contract);

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

nn_real nn_train_action_scorer(NeuralNet *net,
                               const nn_real *candidate_inputs,
                               size_t candidate_count,
                               size_t target_index,
                               nn_real learning_rate);
nn_real nn_train_pairwise_action_ranker(NeuralNet *net,
                                        const nn_real *candidate_inputs,
                                        size_t candidate_count,
                                        size_t target_index,
                                        nn_real margin,
                                        nn_real learning_rate);
nn_real nn_train_softmax_scorer(NeuralNet *net,
                                const nn_real *candidate_inputs,
                                size_t candidate_count,
                                size_t target_index,
                                nn_real learning_rate);
nn_real nn_train_softmax_scorer_weighted(NeuralNet *net,
                                         const nn_real *candidate_inputs,
                                         size_t candidate_count,
                                         size_t selected_index,
                                         nn_real weight,
                                         nn_real learning_rate);
int nn_train_softmax_scorer_weighted_guarded(
    NeuralNet *net,
    const nn_real *candidate_inputs,
    size_t candidate_count,
    size_t selected_index,
    nn_real weight,
    nn_real learning_rate,
    const NNNumericGuardConfig *guard,
    NNTrainingStepCertificate *certificate);
/*
 * Reuses activations from the immediately preceding successful nn_forward_batch
 * or nn_score_action_rows call on the same network and candidate_count. Returns
 * NAN if that cache is no longer valid because another forward/training call
 * intervened.
 */
nn_real nn_train_softmax_scorer_cached(NeuralNet *net,
                                       size_t candidate_count,
                                       size_t selected_index,
                                       nn_real weight,
                                       nn_real learning_rate);
/*
 * Reuses the same cached scorer workspace, but optimizes a policy-style
 * objective:
 *
 *   policy_weight * CE(selected action)
 *   + reference_weight * CE(reference_probabilities)
 *   + entropy_weight * sum_i p_i log(p_i)
 *
 * Positive entropy_weight therefore rewards higher-entropy policies.
 */
nn_real nn_train_softmax_scorer_policy_regularized_cached(
    NeuralNet *net,
    size_t candidate_count,
    size_t selected_index,
    nn_real policy_weight,
    const nn_real *reference_probabilities,
    nn_real reference_weight,
    nn_real entropy_weight,
    nn_real learning_rate);
int nn_train_softmax_scorer_policy_regularized_guarded(
    NeuralNet *net,
    const nn_real *candidate_inputs,
    size_t candidate_count,
    size_t selected_index,
    nn_real policy_weight,
    const nn_real *reference_probabilities,
    nn_real reference_weight,
    nn_real entropy_weight,
    nn_real learning_rate,
    const NNNumericGuardConfig *guard,
    NNTrainingStepCertificate *certificate);

#endif
