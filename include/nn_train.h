#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include "nn_core.h"

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

#endif
