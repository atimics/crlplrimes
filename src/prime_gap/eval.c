/* Policy sampling, calibration metrics, model evaluation, traces, and frequency baselines. */

#include "internal.h"

size_t sample_action(const nn_real *probabilities, uint32_t *rng)
{
    double draw = rng_uniform(rng);
    double cumulative = 0.0;

    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        cumulative += (double)probabilities[i];
        if (draw <= cumulative) {
            return i;
        }
    }

    return ACTION_COUNT - 1;
}

void softmax_temperature(const nn_real *scores,
                         nn_real *probabilities,
                         size_t count,
                         double temperature)
{
    if (temperature <= 0.0) {
        temperature = 1.0;
    }

    if (scores == NULL || probabilities == NULL || count == 0) {
        return;
    }

    double max_score = (double)scores[0] / temperature;
    for (size_t i = 1; i < count; ++i) {
        double scaled = (double)scores[i] / temperature;
        if (scaled > max_score) {
            max_score = scaled;
        }
    }

    double denominator = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double scaled = (double)scores[i] / temperature;
        double probability = exp(scaled - max_score);
        probabilities[i] = (nn_real)probability;
        denominator += probability;
    }

    if (denominator <= 0.0) {
        nn_real uniform = (nn_real)(1.0 / (double)count);
        for (size_t i = 0; i < count; ++i) {
            probabilities[i] = uniform;
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        probabilities[i] = (nn_real)((double)probabilities[i] / denominator);
    }
}

static double safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

double distribution_entropy(const nn_real *probabilities, size_t count)
{
    double entropy = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double probability = safe_probability((double)probabilities[i]);
        entropy -= probability * log(probability);
    }
    return entropy;
}

double distribution_kl(const nn_real *reference,
                       const nn_real *probabilities,
                       size_t count)
{
    double kl = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double reference_probability = safe_probability((double)reference[i]);
        double probability = safe_probability((double)probabilities[i]);
        kl += reference_probability * (log(reference_probability) - log(probability));
    }
    return kl;
}

size_t eval_sample_index(const GapDataset *dataset, size_t sample, size_t samples)
{
    return samples != dataset->count ? (sample * dataset->count) / samples : sample;
}

static size_t prime_bucket_index(const GapDataset *dataset, uint32_t p)
{
    if (dataset->count <= 1 || dataset->prime[dataset->count - 1] <= dataset->prime[0]) {
        return 0;
    }

    uint64_t numerator = (uint64_t)(p - dataset->prime[0]) * PRIME_BUCKET_COUNT;
    uint64_t denominator = (uint64_t)(dataset->prime[dataset->count - 1] - dataset->prime[0]) + 1u;
    size_t bucket = (size_t)(numerator / denominator);
    return bucket < PRIME_BUCKET_COUNT ? bucket : PRIME_BUCKET_COUNT - 1;
}

static void accumulate_eval_sample(EvalMetrics *metrics,
                                   const Sieve *sieve,
                                   const GapDataset *dataset,
                                   uint32_t p,
                                   uint32_t gap,
                                   const size_t *ranked,
                                   const nn_real *probabilities,
                                   int include_verifier)
{
    size_t target = gap_to_action(gap);
    size_t rank = true_rank(ranked, target);
    int top1_correct = rank == 0;
    double true_probability = probabilities == NULL ? 0.0 : (double)probabilities[target];
    double top1_confidence = probabilities == NULL ? 0.0 : (double)probabilities[ranked[0]];
    size_t calibration_bin =
        top1_confidence >= 1.0 ? CALIBRATION_BINS - 1
                               : (size_t)(top1_confidence * (double)CALIBRATION_BINS);
    size_t bucket = prime_bucket_index(dataset, p);
    VerifierWork direct = direct_scan_work(gap);

    if (calibration_bin >= CALIBRATION_BINS) {
        calibration_bin = CALIBRATION_BINS - 1;
    }

    metrics->top1 += top1_correct ? 1u : 0u;
    metrics->top5 += rank < 5 ? 1u : 0u;
    metrics->top10 += rank < 10 ? 1u : 0u;
    metrics->top1_prime += !action_is_overflow(ranked[0]) &&
                                    is_prime_sieve(sieve, p + (uint32_t)action_gap(ranked[0]))
                                ? 1u
                                : 0u;
    metrics->nll += -log(safe_probability(true_probability));
    metrics->true_action_probability += true_probability;
    metrics->mean_true_rank += (double)rank + 1.0;
    metrics->mean_reciprocal_rank += 1.0 / ((double)rank + 1.0);
    metrics->top1_confidence += top1_confidence;
    metrics->mean_abs_gap_error += gap_error_for_action(ranked[0], gap);
    metrics->direct_scan_exact_tests += (double)direct.exact_tests;
    metrics->proposal_steps_top10 += (double)proposal_steps_with_topk(ranked, 10, target);

    if (top1_correct) {
        metrics->correct_top1_confidence += top1_confidence;
        ++metrics->correct_top1_confidence_count;
    } else {
        metrics->incorrect_top1_confidence += top1_confidence;
        ++metrics->incorrect_top1_confidence_count;
    }

    if (top1_confidence >= 0.50) {
        ++metrics->high_confidence;
        metrics->high_confidence_correct += top1_correct ? 1u : 0u;
    }

    ++metrics->calibration_count[calibration_bin];
    metrics->calibration_confidence[calibration_bin] += top1_confidence;
    metrics->calibration_correct[calibration_bin] += top1_correct ? 1.0 : 0.0;

    ++metrics->bucket_samples[bucket];
    metrics->bucket_top1[bucket] += top1_correct ? 1u : 0u;
    metrics->bucket_top5[bucket] += rank < 5 ? 1u : 0u;
    metrics->bucket_top10[bucket] += rank < 10 ? 1u : 0u;
    metrics->bucket_nll[bucket] += -log(safe_probability(true_probability));
    metrics->bucket_true_rank[bucket] += (double)rank + 1.0;

    if (include_verifier) {
        double verifier_started = now_seconds();
        VerifierWork wheel = wheel_scan_work(sieve, p, gap);
        VerifierWork assisted = assisted_wheel_work(sieve, p, gap, ranked, 10);
        BlockWork block = block_scan_work(sieve, p, gap);
        BlockWork assisted_block = assisted_block_work(sieve, p, gap, ranked, 10);

        ++metrics->verifier_samples;
        metrics->wheel_scan_mod_tests += (double)wheel.mod_tests;
        metrics->wheel_scan_exact_tests += (double)wheel.exact_tests;
        metrics->assisted_wheel_mod_tests_top10 += (double)assisted.mod_tests;
        metrics->assisted_wheel_exact_tests_top10 += (double)assisted.exact_tests;
        metrics->block_scan_cost += block_work_cost(block);
        metrics->assisted_block_cost_top10 += block_work_cost(assisted_block);
        metrics->verifier_seconds += now_seconds() - verifier_started;
    }
}

static void finalize_eval_metrics(EvalMetrics *metrics)
{
    double samples = metrics->samples == 0 ? 1.0 : (double)metrics->samples;

    metrics->nll /= samples;
    metrics->true_action_probability /= samples;
    metrics->mean_true_rank /= samples;
    metrics->mean_reciprocal_rank /= samples;
    metrics->top1_confidence /= samples;
    metrics->mean_abs_gap_error /= samples;
    metrics->direct_scan_exact_tests /= samples;
    metrics->proposal_steps_top10 /= samples;

    if (metrics->correct_top1_confidence_count != 0) {
        metrics->correct_top1_confidence /=
            (double)metrics->correct_top1_confidence_count;
    }
    if (metrics->incorrect_top1_confidence_count != 0) {
        metrics->incorrect_top1_confidence /=
            (double)metrics->incorrect_top1_confidence_count;
    }

    for (size_t bin = 0; bin < CALIBRATION_BINS; ++bin) {
        if (metrics->calibration_count[bin] == 0) {
            continue;
        }

        double bin_count = (double)metrics->calibration_count[bin];
        double confidence = metrics->calibration_confidence[bin] / bin_count;
        double accuracy = metrics->calibration_correct[bin] / bin_count;
        metrics->expected_calibration_error +=
            (bin_count / samples) * fabs(confidence - accuracy);
        metrics->calibration_confidence[bin] = confidence;
        metrics->calibration_correct[bin] = accuracy;
    }

    for (size_t bucket = 0; bucket < PRIME_BUCKET_COUNT; ++bucket) {
        if (metrics->bucket_samples[bucket] == 0) {
            continue;
        }

        double bucket_samples = (double)metrics->bucket_samples[bucket];
        metrics->bucket_nll[bucket] /= bucket_samples;
        metrics->bucket_true_rank[bucket] /= bucket_samples;
    }

    if (metrics->verifier_samples != 0) {
        double verifier_samples = (double)metrics->verifier_samples;
        metrics->wheel_scan_mod_tests /= verifier_samples;
        metrics->wheel_scan_exact_tests /= verifier_samples;
        metrics->assisted_wheel_mod_tests_top10 /= verifier_samples;
        metrics->assisted_wheel_exact_tests_top10 /= verifier_samples;
        metrics->block_scan_cost /= verifier_samples;
        metrics->assisted_block_cost_top10 /= verifier_samples;
    }
}

static int symbolic_rules_reject_action(const SymbolicRuleSet *rules, uint32_t p, size_t action)
{
    if (rules == NULL || action_is_overflow(action)) {
        return 0;
    }

    uint32_t candidate = p + (uint32_t)action_gap(action);
    for (size_t i = 0; i < rules->count; ++i) {
        if (symbolic_rule_rejects(&rules->items[i], candidate)) {
            return 1;
        }
    }
    return 0;
}

static int expanded_rules_reject_action(const ExpandedRuleSet *rules, uint32_t p, size_t action)
{
    if (rules == NULL || action_is_overflow(action)) {
        return 0;
    }

    uint32_t gap = (uint32_t)action_gap(action);
    VerifierWork work = {0, 0, 0};
    for (size_t i = 0; i < rules->count; ++i) {
        if (expanded_rule_rejects(&rules->items[i], p, gap, &work)) {
            return 1;
        }
    }
    return 0;
}

size_t fill_symbolic_survivor_candidates(const SymbolicRuleSet *rules,
                                         nn_real *candidate_inputs,
                                         uint32_t p,
                                         double log_scale,
                                         const FeatureMask *mask,
                                         size_t *survivor_actions)
{
    size_t survivor_count = 0;
    fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (symbolic_rules_reject_action(rules, p, action)) {
            continue;
        }

        if (survivor_count != action) {
            memmove(&candidate_inputs[survivor_count * FEATURE_COUNT],
                    &candidate_inputs[action * FEATURE_COUNT],
                    FEATURE_COUNT * sizeof(*candidate_inputs));
        }
        survivor_actions[survivor_count++] = action;
    }

    return survivor_count;
}

size_t fill_expanded_survivor_candidates(const ExpandedRuleSet *rules,
                                         nn_real *candidate_inputs,
                                         uint32_t p,
                                         double log_scale,
                                         const FeatureMask *mask,
                                         size_t *survivor_actions)
{
    size_t survivor_count = 0;
    fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (expanded_rules_reject_action(rules, p, action)) {
            continue;
        }

        if (survivor_count != action) {
            memmove(&candidate_inputs[survivor_count * FEATURE_COUNT],
                    &candidate_inputs[action * FEATURE_COUNT],
                    FEATURE_COUNT * sizeof(*candidate_inputs));
        }
        survivor_actions[survivor_count++] = action;
    }

    return survivor_count;
}

size_t survivor_index_for_action(const size_t *survivor_actions,
                                 size_t survivor_count,
                                 size_t action)
{
    for (size_t i = 0; i < survivor_count; ++i) {
        if (survivor_actions[i] == action) {
            return i;
        }
    }
    return survivor_count;
}

static void expand_symbolic_probabilities(const size_t *survivor_actions,
                                          size_t survivor_count,
                                          const nn_real *compact_probabilities,
                                          nn_real *probabilities)
{
    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] = (nn_real)0;
    }
    for (size_t i = 0; i < survivor_count; ++i) {
        probabilities[survivor_actions[i]] = compact_probabilities[i];
    }
}

static void expand_symbolic_ranking(const size_t *survivor_actions,
                                    size_t survivor_count,
                                    const size_t *compact_ranked,
                                    size_t *ranked)
{
    uint8_t present[ACTION_COUNT] = {0};
    size_t out = 0;

    for (size_t i = 0; i < survivor_count; ++i) {
        size_t action = survivor_actions[compact_ranked[i]];
        ranked[out++] = action;
        present[action] = 1;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (!present[action]) {
            ranked[out++] = action;
        }
    }
}

EvalMetrics evaluate_model(NeuralNet *net,
                           const Sieve *sieve,
                           const GapDataset *dataset,
                           size_t max_states,
                           double log_scale,
                           const FeatureMask *mask,
                           int include_verifier,
                           double temperature,
                           nn_real *candidate_inputs,
                           nn_real *scores,
                           nn_real *probabilities)
{
    EvalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    size_t ranked[ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];

        double scorer_started = now_seconds();
        fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);
        if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, NULL)) {
            continue;
        }
        softmax_temperature(scores, probabilities, ACTION_COUNT, temperature);
        rank_actions(scores, ranked);
        metrics.scorer_seconds += now_seconds() - scorer_started;
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               p,
                               gap,
                               ranked,
                               probabilities,
                               include_verifier);
    }

    metrics.samples = samples;
    finalize_eval_metrics(&metrics);
    return metrics;
}

EvalMetrics evaluate_model_symbolic_masked(NeuralNet *net,
                                           const Sieve *sieve,
                                           const SymbolicRuleSet *rules,
                                           const GapDataset *dataset,
                                           size_t max_states,
                                           double log_scale,
                                           const FeatureMask *mask,
                                           int include_verifier,
                                           double temperature,
                                           nn_real *candidate_inputs,
                                           nn_real *scores,
                                           nn_real *probabilities)
{
    EvalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    size_t survivor_actions[ACTION_COUNT];
    size_t compact_ranked[ACTION_COUNT];
    size_t ranked[ACTION_COUNT];
    nn_real compact_probabilities[ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];
        double scorer_started = now_seconds();
        size_t survivor_count = fill_symbolic_survivor_candidates(rules,
                                                                  candidate_inputs,
                                                                  p,
                                                                  log_scale,
                                                                  mask,
                                                                  survivor_actions);

        if (survivor_count == 0 ||
            !nn_score_action_rows(net, candidate_inputs, survivor_count, scores, NULL)) {
            continue;
        }
        softmax_temperature(scores, compact_probabilities, survivor_count, temperature);
        nn_rank_action_scores(scores, survivor_count, compact_ranked);
        expand_symbolic_probabilities(survivor_actions,
                                      survivor_count,
                                      compact_probabilities,
                                      probabilities);
        expand_symbolic_ranking(survivor_actions, survivor_count, compact_ranked, ranked);
        metrics.scorer_seconds += now_seconds() - scorer_started;
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               p,
                               gap,
                               ranked,
                               probabilities,
                               include_verifier);
    }

    metrics.samples = samples;
    finalize_eval_metrics(&metrics);
    return metrics;
}

EvalMetrics evaluate_model_expanded_masked(NeuralNet *net,
                                           const Sieve *sieve,
                                           const ExpandedRuleSet *rules,
                                           const GapDataset *dataset,
                                           size_t max_states,
                                           double log_scale,
                                           const FeatureMask *mask,
                                           int include_verifier,
                                           double temperature,
                                           nn_real *candidate_inputs,
                                           nn_real *scores,
                                           nn_real *probabilities)
{
    EvalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    size_t survivor_actions[ACTION_COUNT];
    size_t compact_ranked[ACTION_COUNT];
    size_t ranked[ACTION_COUNT];
    nn_real compact_probabilities[ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];
        double scorer_started = now_seconds();
        size_t survivor_count = fill_expanded_survivor_candidates(rules,
                                                                  candidate_inputs,
                                                                  p,
                                                                  log_scale,
                                                                  mask,
                                                                  survivor_actions);

        if (survivor_count == 0 ||
            !nn_score_action_rows(net, candidate_inputs, survivor_count, scores, NULL)) {
            continue;
        }
        softmax_temperature(scores, compact_probabilities, survivor_count, temperature);
        nn_rank_action_scores(scores, survivor_count, compact_ranked);
        expand_symbolic_probabilities(survivor_actions,
                                      survivor_count,
                                      compact_probabilities,
                                      probabilities);
        expand_symbolic_ranking(survivor_actions, survivor_count, compact_ranked, ranked);
        metrics.scorer_seconds += now_seconds() - scorer_started;
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               p,
                               gap,
                               ranked,
                               probabilities,
                               include_verifier);
    }

    metrics.samples = samples;
    finalize_eval_metrics(&metrics);
    return metrics;
}

EvalMetrics evaluate_model_feature_wheel_masked(NeuralNet *net,
                                                const Sieve *sieve,
                                                const GapDataset *dataset,
                                                size_t max_states,
                                                double log_scale,
                                                const FeatureMask *mask,
                                                int include_verifier,
                                                double temperature,
                                                nn_real *candidate_inputs,
                                                nn_real *scores,
                                                nn_real *probabilities)
{
    EvalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    size_t ranked[ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];

        double scorer_started = now_seconds();
        fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);
        if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, NULL)) {
            continue;
        }
        softmax_temperature(scores, probabilities, ACTION_COUNT, temperature);
        apply_feature_wheel_filter(p, scores, probabilities);
        rank_actions(scores, ranked);
        metrics.scorer_seconds += now_seconds() - scorer_started;
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               p,
                               gap,
                               ranked,
                               probabilities,
                               include_verifier);
    }

    metrics.samples = samples;
    finalize_eval_metrics(&metrics);
    return metrics;
}

static double temperature_nll(NeuralNet *net,
                              const GapDataset *dataset,
                              size_t max_states,
                              double log_scale,
                              const FeatureMask *mask,
                              double temperature,
                              nn_real *candidate_inputs,
                              nn_real *scores,
                              nn_real *probabilities)
{
    size_t samples = dataset->count;
    double nll = 0.0;

    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    if (samples == 0) {
        return INFINITY;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        size_t target = gap_to_action(dataset->gap[index]);

        fill_candidate_features_masked(candidate_inputs,
                                       dataset->prime[index],
                                       log_scale,
                                       mask);
        if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, NULL)) {
            continue;
        }
        softmax_temperature(scores, probabilities, ACTION_COUNT, temperature);
        nll += -log(safe_probability((double)probabilities[target]));
    }

    return nll / (double)samples;
}

static double temperature_nll_symbolic_masked(NeuralNet *net,
                                             const SymbolicRuleSet *rules,
                                             const GapDataset *dataset,
                                             size_t max_states,
                                             double log_scale,
                                             const FeatureMask *mask,
                                             double temperature,
                                             nn_real *candidate_inputs,
                                             nn_real *scores,
                                             nn_real *probabilities)
{
    size_t samples = dataset->count;
    double nll = 0.0;
    size_t survivor_actions[ACTION_COUNT];

    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }
    if (samples == 0) {
        return INFINITY;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        size_t target = gap_to_action(dataset->gap[index]);
        size_t survivor_count = fill_symbolic_survivor_candidates(rules,
                                                                  candidate_inputs,
                                                                  dataset->prime[index],
                                                                  log_scale,
                                                                  mask,
                                                                  survivor_actions);
        size_t survivor_target =
            survivor_index_for_action(survivor_actions, survivor_count, target);

        if (survivor_count == 0 || survivor_target == survivor_count ||
            !nn_score_action_rows(net, candidate_inputs, survivor_count, scores, NULL)) {
            nll += -log(safe_probability(0.0));
            continue;
        }
        softmax_temperature(scores, probabilities, survivor_count, temperature);
        nll += -log(safe_probability((double)probabilities[survivor_target]));
    }

    return nll / (double)samples;
}

double fit_temperature(NeuralNet *net,
                       const GapDataset *dataset,
                       size_t max_states,
                       double log_scale,
                       const FeatureMask *mask,
                       nn_real *candidate_inputs,
                       nn_real *scores,
                       nn_real *probabilities,
                       double *best_nll)
{
    static const double candidates[] = {
        0.25, 0.35, 0.50, 0.70, 0.85, 1.00, 1.20, 1.50,
        2.00, 2.50, 3.00, 4.00, 6.00, 8.00, 12.00,
    };
    double best_temperature = 1.0;
    double lowest_nll = INFINITY;

    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        double candidate_nll = temperature_nll(net,
                                               dataset,
                                               max_states,
                                               log_scale,
                                               mask,
                                               candidates[i],
                                               candidate_inputs,
                                               scores,
                                               probabilities);
        if (candidate_nll < lowest_nll) {
            lowest_nll = candidate_nll;
            best_temperature = candidates[i];
        }
    }

    if (best_nll != NULL) {
        *best_nll = lowest_nll;
    }
    return best_temperature;
}

double fit_temperature_symbolic_masked(NeuralNet *net,
                                       const SymbolicRuleSet *rules,
                                       const GapDataset *dataset,
                                       size_t max_states,
                                       double log_scale,
                                       const FeatureMask *mask,
                                       nn_real *candidate_inputs,
                                       nn_real *scores,
                                       nn_real *probabilities,
                                       double *best_nll)
{
    static const double candidates[] = {
        0.25, 0.35, 0.50, 0.70, 0.85, 1.00, 1.20, 1.50,
        2.00, 2.50, 3.00, 4.00, 6.00, 8.00, 12.00,
    };
    double best_temperature = 1.0;
    double lowest_nll = INFINITY;

    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        double candidate_nll = temperature_nll_symbolic_masked(net,
                                                               rules,
                                                               dataset,
                                                               max_states,
                                                               log_scale,
                                                               mask,
                                                               candidates[i],
                                                               candidate_inputs,
                                                               scores,
                                                               probabilities);
        if (candidate_nll < lowest_nll) {
            lowest_nll = candidate_nll;
            best_temperature = candidates[i];
        }
    }

    if (best_nll != NULL) {
        *best_nll = lowest_nll;
    }
    return best_temperature;
}

EvalMetrics evaluate_frequency_policy(const Sieve *sieve,
                                      const GapDataset *dataset,
                                      const nn_real *base_probabilities,
                                      size_t max_states,
                                      int include_verifier,
                                      int apply_wheel_filter)
{
    EvalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    size_t ranked[ACTION_COUNT];
    nn_real scores[ACTION_COUNT];
    nn_real probabilities[ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        double scorer_started = now_seconds();
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            scores[action] = base_probabilities[action];
            probabilities[action] = base_probabilities[action];
        }
        if (apply_wheel_filter) {
            apply_feature_wheel_filter(dataset->prime[index], scores, probabilities);
        }
        rank_actions(scores, ranked);
        metrics.scorer_seconds += now_seconds() - scorer_started;
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               dataset->prime[index],
                               dataset->gap[index],
                               ranked,
                               probabilities,
                               include_verifier);
    }

    metrics.samples = samples;
    finalize_eval_metrics(&metrics);
    return metrics;
}

static void print_reliability_bins(const EvalMetrics *metrics)
{
    printf("    reliability bins");
    for (size_t bin = 0; bin < CALIBRATION_BINS; ++bin) {
        if (metrics->calibration_count[bin] == 0) {
            continue;
        }
        double low = (double)bin / (double)CALIBRATION_BINS;
        double high = (double)(bin + 1) / (double)CALIBRATION_BINS;
        printf(" [%.1f,%.1f):n=%zu conf=%.3f acc=%.3f",
               low,
               high,
               metrics->calibration_count[bin],
               metrics->calibration_confidence[bin],
               metrics->calibration_correct[bin]);
    }
    printf("\n");
}

void print_metrics(const char *label, EvalMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    double scorer_ms = 1000.0 * metrics.scorer_seconds / samples;
    double verifier_ms = metrics.verifier_samples == 0
                             ? 0.0
                             : 1000.0 * metrics.verifier_seconds /
                                   (double)metrics.verifier_samples;
    double high_confidence_coverage =
        100.0 * (double)metrics.high_confidence / samples;
    double high_confidence_accuracy =
        metrics.high_confidence == 0
            ? 0.0
            : 100.0 * (double)metrics.high_confidence_correct /
                  (double)metrics.high_confidence;
    double proposal_saved = metrics.direct_scan_exact_tests - metrics.proposal_steps_top10;
    double proposal_saved_pct = metrics.direct_scan_exact_tests > 0.0
                                    ? 100.0 * proposal_saved / metrics.direct_scan_exact_tests
                                    : 0.0;
    double wheel_exact_saved = metrics.direct_scan_exact_tests - metrics.wheel_scan_exact_tests;
    double wheel_exact_saved_pct = metrics.direct_scan_exact_tests > 0.0
                                       ? 100.0 * wheel_exact_saved /
                                             metrics.direct_scan_exact_tests
                                       : 0.0;
    double assisted_exact_saved =
        metrics.wheel_scan_exact_tests - metrics.assisted_wheel_exact_tests_top10;
    double assisted_exact_saved_pct = metrics.wheel_scan_exact_tests > 0.0
                                          ? 100.0 * assisted_exact_saved /
                                                metrics.wheel_scan_exact_tests
                                          : 0.0;
    double block_delta = metrics.block_scan_cost - metrics.assisted_block_cost_top10;
    double block_delta_pct = metrics.block_scan_cost > 0.0
                                 ? 100.0 * block_delta / metrics.block_scan_cost
                                 : 0.0;

    printf("\n[%s]\n", label);
    printf("  capability\n");
    printf("    samples=%zu top1=%.2f%% top5=%.2f%% top10=%.2f%% top1_prime=%.2f%%\n",
           metrics.samples,
           100.0 * (double)metrics.top1 / samples,
           100.0 * (double)metrics.top5 / samples,
           100.0 * (double)metrics.top10 / samples,
           100.0 * (double)metrics.top1_prime / samples);
    printf("    nll=%.4f p_true=%.4f mean_rank=%.2f mrr=%.4f mae_gap=%.2f\n",
           metrics.nll,
           metrics.true_action_probability,
           metrics.mean_true_rank,
           metrics.mean_reciprocal_rank,
           metrics.mean_abs_gap_error);
    printf("    proposal_steps baseline=%.2f model_top10=%.2f saved=%.2f (%.1f%%)\n",
           metrics.direct_scan_exact_tests,
           metrics.proposal_steps_top10,
           proposal_saved,
           proposal_saved_pct);

    printf("  calibration\n");
    printf("    top1_conf=%.4f correct_conf=%.4f incorrect_conf=%.4f ece=%.4f\n",
           metrics.top1_confidence,
           metrics.correct_top1_confidence,
           metrics.incorrect_top1_confidence,
           metrics.expected_calibration_error);
    printf("    conf>=0.50 coverage=%.2f%% accuracy=%.2f%%\n",
           high_confidence_coverage,
           high_confidence_accuracy);
    print_reliability_bins(&metrics);

    printf("  wall-clock cost\n");
    printf("    scorer=%.6f ms/sample total=%.6fs verifier=%.6f ms/sample total=%.6fs\n",
           scorer_ms,
           metrics.scorer_seconds,
           verifier_ms,
           metrics.verifier_seconds);

    printf("  prime buckets\n");
    for (size_t bucket = 0; bucket < PRIME_BUCKET_COUNT; ++bucket) {
        double bucket_samples =
            metrics.bucket_samples[bucket] == 0 ? 1.0 : (double)metrics.bucket_samples[bucket];
        printf("    q%zu samples=%zu top1=%.2f%% top5=%.2f%% top10=%.2f%% "
               "nll=%.4f mean_rank=%.2f\n",
               bucket + 1,
               metrics.bucket_samples[bucket],
               100.0 * (double)metrics.bucket_top1[bucket] / bucket_samples,
               100.0 * (double)metrics.bucket_top5[bucket] / bucket_samples,
               100.0 * (double)metrics.bucket_top10[bucket] / bucket_samples,
               metrics.bucket_nll[bucket],
               metrics.bucket_true_rank[bucket]);
    }

    if (metrics.verifier_samples == 0) {
        printf("  verifier diagnostics\n");
        printf("    skipped for this capability-only run\n");
        return;
    }

    printf("  verifier diagnostics\n");
    printf("    wheel exact %.2f -> %.2f saved=%.2f (%.1f%%) wheel_mods=%.2f\n",
           metrics.direct_scan_exact_tests,
           metrics.wheel_scan_exact_tests,
           wheel_exact_saved,
           wheel_exact_saved_pct,
           metrics.wheel_scan_mod_tests);
    printf("    assisted_wheel exact=%.2f delta=%.2f (%.1f%%) mods=%.2f\n",
           metrics.assisted_wheel_exact_tests_top10,
           assisted_exact_saved,
           assisted_exact_saved_pct,
           metrics.assisted_wheel_mod_tests_top10);
    printf("    block cost=%.2f assisted_block=%.2f delta=%.2f (%.1f%%)\n",
           metrics.block_scan_cost,
           metrics.assisted_block_cost_top10,
           block_delta,
           block_delta_pct);
}

void print_comparison_header(void)
{
    printf("\n=== Policy Comparison ===\n");
    printf("%-30s %6s %8s %8s %8s %8s %9s %9s %9s\n",
           "model",
           "temp",
           "top1",
           "top10",
           "nll",
           "ece",
           "conf",
           "hi_cov",
           "hi_acc");
}

void print_comparison_row(const char *label, double temperature, EvalMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    double high_confidence_coverage =
        100.0 * (double)metrics.high_confidence / samples;
    double high_confidence_accuracy =
        metrics.high_confidence == 0
            ? 0.0
            : 100.0 * (double)metrics.high_confidence_correct /
                  (double)metrics.high_confidence;

    printf("%-30s %6.2f %7.2f%% %7.2f%% %8.4f %8.4f %9.4f %8.2f%% %8.2f%%\n",
           label,
           temperature,
           100.0 * (double)metrics.top1 / samples,
           100.0 * (double)metrics.top10 / samples,
           metrics.nll,
           metrics.expected_calibration_error,
           metrics.top1_confidence,
           high_confidence_coverage,
           high_confidence_accuracy);
}

static uint64_t prime_state_hash(uint32_t p)
{
    uint64_t hash = 1469598103934665603ull;
    hash ^= (uint64_t)p;
    hash *= 1099511628211ull;
    return hash;
}

static uint32_t prime_action_id_for_action(size_t action)
{
    return (uint32_t)action;
}

int prime_action_from_action_id(uint32_t action_id, size_t *out_action)
{
    if (action_id >= ACTION_COUNT || out_action == NULL) {
        return 0;
    }

    *out_action = (size_t)action_id;
    return 1;
}

void fill_action_ids(uint32_t *action_ids)
{
    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        action_ids[action] = prime_action_id_for_action(action);
    }
}

size_t action_row_for_action_id(const uint32_t *action_ids,
                                size_t candidate_count,
                                uint32_t action_id)
{
    for (size_t row = 0; row < candidate_count; ++row) {
        if (action_ids[row] == action_id) {
            return row;
        }
    }
    return candidate_count;
}

static size_t action_row_for_action(const uint32_t *action_ids,
                                    size_t candidate_count,
                                    size_t action)
{
    return action_row_for_action_id(action_ids,
                                    candidate_count,
                                    prime_action_id_for_action(action));
}

int save_prime_checkpoint_with_meta(NeuralNet *net,
                                    const char *path,
                                    const char *notes,
                                    const char *parent_hash,
                                    const char *rule_hash,
                                    char out_hash[65])
{
    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = 1;
    snprintf(meta.parent_model_hash,
             sizeof(meta.parent_model_hash),
             "%s",
             parent_hash == NULL ? "unknown" : parent_hash);
    if (rule_hash != NULL && rule_hash[0] != '\0' && strcmp(rule_hash, "unknown") != 0) {
        snprintf(meta.training_trace_root, sizeof(meta.training_trace_root), "%s", rule_hash);
    }
    snprintf(meta.notes,
             sizeof(meta.notes),
             "%s",
             notes == NULL ? "prime_gap action scorer" : notes);

    if (!nn_save_checkpoint(net, path, &meta) || !nn_hash_checkpoint_hex(path, out_hash)) {
        return 0;
    }
    printf("saved checkpoint: %s hash=%s\n", path, out_hash);
    return 1;
}

int save_prime_checkpoint(NeuralNet *net, const char *path, char out_hash[65])
{
    return save_prime_checkpoint_with_meta(net,
                                           path,
                                           "prime_gap action scorer",
                                           "unknown",
                                           NULL,
                                           out_hash);
}

int emit_prime_gap_trace(NeuralNet *net,
                         const Sieve *sieve,
                         const GapDataset *dataset,
                         size_t max_states,
                         double log_scale,
                         const FeatureMask *mask,
                         const char *path,
                         const char *model_hash,
                         const char *rule_hash,
                         uint64_t seed,
                         nn_real *candidate_inputs,
                         nn_real *scores,
                         nn_real *probabilities)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    uint32_t action_ids[ACTION_COUNT];
    fill_action_ids(action_ids);

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    int ok = 1;
    for (size_t sample = 0; ok && sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];
        size_t target = gap_to_action(gap);
        size_t target_index = action_row_for_action(action_ids, ACTION_COUNT, target);
        size_t ranked[ACTION_COUNT];

        if (target_index == ACTION_COUNT) {
            ok = 0;
            break;
        }
        fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);
        ok = nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, probabilities) &&
             nn_rank_action_scores(scores, ACTION_COUNT, ranked);
        if (!ok) {
            break;
        }

        VerifierWork wheel = wheel_scan_work(sieve, p, gap);
        VerifierWork assisted = assisted_wheel_work(sieve, p, gap, ranked, 10);
        BlockWork block = block_scan_work(sieve, p, gap);
        BlockWork assisted_block = assisted_block_work(sieve, p, gap, ranked, 10);
        size_t proposal_steps = proposal_steps_with_topk(ranked, 10, target);
        char verifier_cost[256];
        char extra[640];

        snprintf(verifier_cost,
                 sizeof(verifier_cost),
                 "{\"proposal_steps\":%zu,\"wheel_exact\":%zu,"
                 "\"assisted_wheel\":%zu,\"block_cost\":%.6f,"
                 "\"assisted_block\":%.6f}",
                 proposal_steps,
                 wheel.exact_tests,
                 assisted.exact_tests,
                 block_work_cost(block),
                 block_work_cost(assisted_block));
        snprintf(extra,
                 sizeof(extra),
                 "\"p\":%u,\"true_gap\":%u,\"proposal_steps\":%zu,"
                 "\"wheel_exact\":%zu,\"assisted_wheel\":%zu,"
                 "\"block_cost\":%.6f,\"assisted_block\":%.6f,\"log_scale\":%.17g,"
                 "\"rule_set_hash\":\"%s\",\"feature_set\":\"%s\"",
                 p,
                 gap,
                 proposal_steps,
                 wheel.exact_tests,
                 assisted.exact_tests,
                 block_work_cost(block),
                 block_work_cost(assisted_block),
                 log_scale,
                 rule_hash == NULL ? "unknown" : rule_hash,
                 mask == NULL ? FEATURE_MASK_MINIMAL.name : mask->name);

        NNTraceRecord record;
        memset(&record, 0, sizeof(record));
        record.task_name = "prime_gap";
        record.model_hash_hex = model_hash == NULL ? "unknown" : model_hash;
        record.state_hash = prime_state_hash(p);
        record.feature_encoder_version = 1;
        record.candidate_count = ACTION_COUNT;
        record.action_ids = action_ids;
        record.scores = scores;
        record.probabilities = probabilities;
        record.selected_index = ranked[0];
        record.target_index = target_index;
        record.loss = (nn_real)(-log(safe_probability((double)probabilities[target_index])));
        record.verifier_result =
            action_ids[ranked[0]] == action_ids[target_index] ? "correct" : "corrected";
        record.verifier_cost_json = verifier_cost;
        record.seed = seed + (uint64_t)sample;
        record.extra_json = extra;
        ok = nn_trace_write_jsonl(fp, &record);
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (ok) {
        printf("wrote trace: %s rows=%zu model_hash=%s\n",
               path,
               samples,
               model_hash == NULL ? "unknown" : model_hash);
    }
    return ok;
}

void build_frequency_policy(const GapDataset *dataset,
                            size_t *ranked,
                            nn_real *probabilities)
{
    size_t counts[ACTION_COUNT] = {0};
    nn_real scores[ACTION_COUNT];
    double denominator = (double)dataset->count + (double)ACTION_COUNT;

    for (size_t i = 0; i < dataset->count; ++i) {
        ++counts[gap_to_action(dataset->gap[i])];
    }

    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        double probability = ((double)counts[i] + 1.0) / denominator;
        probabilities[i] = (nn_real)probability;
        scores[i] = probabilities[i];
    }

    rank_actions(scores, ranked);
}

void free_conditioned_frequency_policy(ConditionedFrequencyPolicy *policy)
{
    if (policy == NULL) {
        return;
    }

    free(policy->action_counts);
    memset(policy, 0, sizeof(*policy));
}

int build_conditioned_frequency_policy(ConditionedFrequencyPolicy *policy,
                                       const GapDataset *dataset)
{
    memset(policy, 0, sizeof(*policy));
    policy->action_counts = calloc((size_t)CONDITION_SIGNATURE_COUNT * ACTION_COUNT,
                                   sizeof(*policy->action_counts));
    if (policy->action_counts == NULL) {
        return 0;
    }

    for (size_t i = 0; i < dataset->count; ++i) {
        size_t action = gap_to_action(dataset->gap[i]);
        size_t signature = residue_signature(dataset->prime[i],
                                             action,
                                             CONDITION_FEATURE_MOD_COUNT);
        ++policy->action_counts[signature * ACTION_COUNT + action];
        ++policy->condition_totals[signature];
    }

    return 1;
}

void conditioned_frequency_scores(const ConditionedFrequencyPolicy *policy,
                                  uint32_t p,
                                  nn_real *scores,
                                  nn_real *probabilities)
{
    double sum = 0.0;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        size_t signature = residue_signature(p, action, CONDITION_FEATURE_MOD_COUNT);
        size_t count = policy->action_counts[signature * ACTION_COUNT + action];
        size_t total = policy->condition_totals[signature];
        double score = ((double)count + 1.0) / ((double)total + (double)ACTION_COUNT);
        scores[action] = (nn_real)score;
        probabilities[action] = (nn_real)score;
        sum += score;
    }

    if (sum <= 0.0) {
        nn_real uniform = (nn_real)(1.0 / (double)ACTION_COUNT);
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = uniform;
        }
        return;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] = (nn_real)((double)probabilities[action] / sum);
    }
}

EvalMetrics evaluate_conditioned_frequency_policy(
    const ConditionedFrequencyPolicy *policy,
    const Sieve *sieve,
    const GapDataset *dataset,
    size_t max_states,
    int include_verifier,
    int apply_wheel_filter,
    nn_real *scores,
    nn_real *probabilities)
{
    EvalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    size_t ranked[ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        double scorer_started = now_seconds();
        conditioned_frequency_scores(policy, dataset->prime[index], scores, probabilities);
        if (apply_wheel_filter) {
            apply_feature_wheel_filter(dataset->prime[index], scores, probabilities);
        }
        rank_actions(scores, ranked);
        metrics.scorer_seconds += now_seconds() - scorer_started;
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               dataset->prime[index],
                               dataset->gap[index],
                               ranked,
                               probabilities,
                               include_verifier);
    }

    metrics.samples = samples;
    finalize_eval_metrics(&metrics);
    return metrics;
}
