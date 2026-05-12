#include "prime_gap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int train_supervised(NeuralNet *net,
                            const char *label,
                            const GapDataset *dataset,
                            size_t max_states,
                            size_t epochs,
                            nn_real learning_rate,
                            double log_scale,
                            const FeatureMask *mask,
                            nn_real entropy_weight,
                            int wheel_survivors_only,
                            nn_real *candidate_inputs,
                            size_t *order,
                            uint32_t *rng)
{
    size_t train_steps = dataset->count;
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double loss = 0.0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            size_t target_action = gap_to_action(dataset->gap[index]);
            size_t target_index = target_action;
            size_t candidate_count = ACTION_COUNT;

            if (wheel_survivors_only) {
                candidate_count = fill_wheel_survivor_features(candidate_inputs,
                                                               dataset->prime[index],
                                                               target_action,
                                                               log_scale,
                                                               mask,
                                                               &target_index);
                if (candidate_count == 0 || target_index >= candidate_count) {
                    return 0;
                }
            } else {
                fill_candidate_features_masked(
                    candidate_inputs, dataset->prime[index], log_scale, mask);
            }

            double profiled = prime_gap_profile_start();
            nn_real step_loss = entropy_weight > (nn_real)0
                                    ? nn_train_softmax_scorer_entropy(net,
                                                                      candidate_inputs,
                                                                      candidate_count,
                                                                      target_index,
                                                                      entropy_weight,
                                                                      learning_rate)
                                    : nn_train_softmax_scorer(net,
                                                              candidate_inputs,
                                                              candidate_count,
                                                              target_index,
                                                              learning_rate);
            prime_gap_profile_stop(PRIME_GAP_PROFILE_TRAIN_UPDATE, profiled);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu loss=%.4f entropy_weight=%.3f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               loss / (double)train_steps,
               (double)entropy_weight,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

int train_interval_supervised(NeuralNet *net,
                                     const char *label,
                                     const Sieve *sieve,
                                     const GapDataset *dataset,
                                     size_t max_states,
                                     size_t candidate_count,
                                     double gap_scale,
                                     size_t epochs,
                                     nn_real learning_rate,
                                     double log_scale,
                                     const FeatureMask *mask,
                                     nn_real *candidate_inputs,
                                     nn_real *targets,
                                     size_t *order,
                                     uint32_t *rng)
{
    size_t train_steps = dataset->count;
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }
    if (candidate_count == 0 || candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        return 0;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double loss = 0.0;
        size_t updates = 0;
        size_t positives = 0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            uint32_t p = dataset->prime[index];
            size_t positive_count = build_interval_prime_targets(sieve, p, candidate_count, targets);
            if (positive_count == 0) {
                continue;
            }

            fill_interval_candidate_features(
                candidate_inputs, p, candidate_count, gap_scale, log_scale, mask);
            double profiled = prime_gap_profile_start();
            nn_real step_loss = nn_train_softmax_scorer_targets(net,
                                                                candidate_inputs,
                                                                candidate_count,
                                                                targets,
                                                                learning_rate);
            prime_gap_profile_stop(PRIME_GAP_PROFILE_TRAIN_UPDATE, profiled);
            if (isnan((double)step_loss)) {
                return 0;
            }

            loss += (double)step_loss;
            positives += positive_count;
            ++updates;
        }

        if (updates == 0) {
            return 0;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu loss=%.4f avg_primes=%.2f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               loss / (double)updates,
               (double)positives / (double)updates,
               updates,
               elapsed > 0.0 ? (double)updates / elapsed : 0.0);
    }

    return 1;
}

int train_strict_verifier_rl(NeuralNet *net,
                                    const char *label,
                                    const GapDataset *dataset,
                                    size_t max_states,
                                    size_t epochs,
                                    nn_real learning_rate,
                                    double log_scale,
                                    const FeatureMask *mask,
                                    nn_real *candidate_inputs,
                                    nn_real *scores,
                                    nn_real *probabilities,
                                    size_t *order,
                                    uint32_t *rng)
{
    size_t train_steps = dataset->count;
    double baseline = 0.0;
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double objective = 0.0;
        size_t exact_hits = 0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            size_t target = gap_to_action(dataset->gap[index]);

            fill_candidate_features_masked(candidate_inputs, dataset->prime[index], log_scale, mask);
            double profiled = prime_gap_profile_start();
            nn_forward_batch(net, candidate_inputs, scores, ACTION_COUNT);
            prime_gap_profile_stop(PRIME_GAP_PROFILE_FORWARD, profiled);

            profiled = prime_gap_profile_start();
            nn_softmax(scores, probabilities, ACTION_COUNT);

            size_t selected = sample_action(probabilities, rng);
            double reward = selected == target ? 1.0 : 0.0;
            double advantage = reward - baseline;

            baseline += 0.01 * (reward - baseline);
            exact_hits += reward > 0.0 ? 1u : 0u;
            prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);

            profiled = prime_gap_profile_start();
            nn_real step_loss = nn_train_softmax_scorer_cached(net,
                                                               ACTION_COUNT,
                                                               selected,
                                                               (nn_real)advantage,
                                                               learning_rate);
            prime_gap_profile_stop(PRIME_GAP_PROFILE_TRAIN_UPDATE, profiled);
            if (isnan((double)step_loss)) {
                return 0;
            }
            objective += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu sampled_exact=%.2f%% baseline=%.3f objective=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               100.0 * (double)exact_hits / (double)train_steps,
               baseline,
               objective / (double)train_steps,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

int run_supervised_ablation(const char *label,
                                   const FeatureMask *mask,
                                   const NNConfig *config,
                                   const Sieve *sieve,
                                   const GapDataset *train,
                                   const GapDataset *test,
                                   size_t max_train_states,
                                   size_t max_test_states,
                                   size_t supervised_epochs,
                                   double log_scale,
                                   nn_real *candidate_inputs,
                                   nn_real *scores,
                                   nn_real *probabilities,
                                   size_t *order,
                                   uint32_t *rng,
                                   EvalMetrics *metrics_out)
{
    NeuralNet *net = nn_create_config(config);
    int ok;

    if (net == NULL) {
        return 0;
    }

    printf("\ntraining %s (%s features)\n", label, mask->name);
    ok = train_supervised(net,
                          label,
                          train,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          mask,
                          (nn_real)0,
                          0,
                          candidate_inputs,
                          order,
                          rng);
    if (ok) {
        EvalMetrics metrics = evaluate_model(net,
                                             sieve,
                                             test,
                                             max_test_states,
                                             log_scale,
                                             mask,
                                             1.0,
                                             0,
                                             0,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        if (metrics_out != NULL) {
            *metrics_out = metrics;
        }
        print_metrics(label, metrics);
    }

    nn_free(net);
    return ok;
}
