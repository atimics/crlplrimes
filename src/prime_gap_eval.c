#include "prime_gap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double TEMPERATURE_GRID[] = {0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00, 6.00, 8.00};

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

double safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

size_t eval_sample_index(const GapDataset *dataset, size_t sample, size_t samples)
{
    return samples != dataset->count ? (sample * dataset->count) / samples : sample;
}

size_t prime_bucket_index(const GapDataset *dataset, uint32_t p)
{
    if (dataset->count <= 1 || dataset->prime[dataset->count - 1] <= dataset->prime[0]) {
        return 0;
    }

    uint64_t numerator = (uint64_t)(p - dataset->prime[0]) * PRIME_BUCKET_COUNT;
    uint64_t denominator = (uint64_t)(dataset->prime[dataset->count - 1] - dataset->prime[0]) + 1u;
    size_t bucket = (size_t)(numerator / denominator);
    return bucket < PRIME_BUCKET_COUNT ? bucket : PRIME_BUCKET_COUNT - 1;
}

void accumulate_eval_sample(EvalMetrics *metrics,
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
    }
}

void finalize_eval_metrics(EvalMetrics *metrics)
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

EvalMetrics evaluate_model(NeuralNet *net,
                                  const Sieve *sieve,
                                  const GapDataset *dataset,
                                  size_t max_states,
                                  double log_scale,
                                  const FeatureMask *mask,
                                  double temperature,
                                  int wheel_survivors_only,
                                  int include_verifier,
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
    size_t completed = 0;
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];

        fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);
        double profiled = prime_gap_profile_start();
        if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, NULL)) {
            prime_gap_profile_stop(PRIME_GAP_PROFILE_FORWARD, profiled);
            break;
        }
        prime_gap_profile_stop(PRIME_GAP_PROFILE_FORWARD, profiled);

        profiled = prime_gap_profile_start();
        if (wheel_survivors_only) {
            softmax_temperature_wheel(scores, probabilities, p, temperature);
            rank_actions_wheel(scores, p, ranked);
        } else {
            softmax_temperature(scores, probabilities, ACTION_COUNT, temperature);
            rank_actions(scores, ranked);
        }
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               p,
                               gap,
                               ranked,
                               probabilities,
                               include_verifier);
        ++completed;
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    metrics.samples = completed;
    metrics.temperature = temperature <= 0.0 ? 1.0 : temperature;
    finalize_eval_metrics(&metrics);
    return metrics;
}

EvalMetrics evaluate_static_policy(const Sieve *sieve,
                                          const GapDataset *dataset,
                                          const size_t *ranked,
                                          const nn_real *probabilities,
                                          size_t max_states,
                                          int include_verifier)
{
    EvalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        double profiled = prime_gap_profile_start();
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               dataset->prime[index],
                               dataset->gap[index],
                               ranked,
                               probabilities,
                               include_verifier);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    metrics.samples = samples;
    metrics.temperature = 1.0;
    finalize_eval_metrics(&metrics);
    return metrics;
}

void fill_interval_candidate_features(nn_real *inputs,
                                             uint32_t p,
                                             size_t candidate_count,
                                             double gap_scale,
                                             double log_scale,
                                             const FeatureMask *mask)
{
    double profiled = prime_gap_profile_start();

    for (size_t action = 0; action < candidate_count; ++action) {
        fill_gap_features(&inputs[action * FEATURE_COUNT],
                          p,
                          indexed_gap(action),
                          0,
                          gap_scale,
                          log_scale,
                          mask);
    }

    prime_gap_profile_stop(PRIME_GAP_PROFILE_FEATURES, profiled);
}

size_t build_interval_prime_targets(const Sieve *sieve,
                                           uint32_t p,
                                           size_t candidate_count,
                                           nn_real *targets)
{
    size_t positive_count = 0;
    double profiled = prime_gap_profile_start();

    for (size_t action = 0; action < candidate_count; ++action) {
        uint32_t candidate = p + indexed_gap(action);
        int positive = is_prime_sieve(sieve, candidate);
        targets[action] = positive ? (nn_real)1 : (nn_real)0;
        positive_count += positive ? 1u : 0u;
    }

    if (positive_count == 0) {
        prime_gap_profile_stop(PRIME_GAP_PROFILE_TARGETS, profiled);
        return 0;
    }

    nn_real target = (nn_real)1 / (nn_real)positive_count;
    for (size_t action = 0; action < candidate_count; ++action) {
        if (targets[action] > (nn_real)0) {
            targets[action] = target;
        }
    }

    prime_gap_profile_stop(PRIME_GAP_PROFILE_TARGETS, profiled);
    return positive_count;
}

void free_interval_target_cache(IntervalTargetCache *cache)
{
    if (cache == NULL) {
        return;
    }

    free(cache->targets);
    free(cache->positive_counts);
    cache->targets = NULL;
    cache->positive_counts = NULL;
    cache->sample_count = 0;
    cache->candidate_count = 0;
}

int build_interval_eval_target_cache(const Sieve *sieve,
                                     const GapDataset *dataset,
                                     size_t max_states,
                                     size_t candidate_count,
                                     IntervalTargetCache *cache)
{
    size_t samples;

    if (cache == NULL || candidate_count == 0 ||
        candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        return 0;
    }

    free_interval_target_cache(cache);
    samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    cache->targets = calloc(samples * candidate_count, sizeof(*cache->targets));
    cache->positive_counts = calloc(samples, sizeof(*cache->positive_counts));
    if (cache->targets == NULL || cache->positive_counts == NULL) {
        free_interval_target_cache(cache);
        return 0;
    }

    cache->sample_count = samples;
    cache->candidate_count = candidate_count;
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        nn_real *targets = &cache->targets[sample * candidate_count];
        cache->positive_counts[sample] =
            build_interval_prime_targets(sieve, dataset->prime[index], candidate_count, targets);
    }

    return 1;
}

void accumulate_interval_sample(IntervalMetrics *metrics,
                                       uint32_t p,
                                       const size_t *ranked,
                                       const nn_real *probabilities,
                                       const nn_real *targets,
                                       size_t candidate_count,
                                       size_t positive_count)
{
    size_t top_limit_5 = candidate_count < 5 ? candidate_count : 5;
    size_t top_limit_10 = candidate_count < 10 ? candidate_count : 10;
    size_t hits = 0;
    size_t hits5 = 0;
    size_t hits10 = 0;
    double average_precision = 0.0;
    double positive_probability = 0.0;
    double nll = 0.0;
    size_t first_rank = candidate_count;

    if (positive_count == 0) {
        return;
    }

    for (size_t action = 0; action < candidate_count; ++action) {
        if (targets[action] <= (nn_real)0) {
            continue;
        }
        positive_probability += (double)probabilities[action];
        nll -= (double)targets[action] * log(safe_probability((double)probabilities[action]));
    }

    for (size_t rank = 0; rank < candidate_count; ++rank) {
        size_t action = ranked[rank];
        if (targets[action] <= (nn_real)0) {
            continue;
        }

        if (first_rank == candidate_count) {
            first_rank = rank;
        }
        ++hits;
        average_precision += (double)hits / ((double)rank + 1.0);
        if (rank < top_limit_5) {
            ++hits5;
        }
        if (rank < top_limit_10) {
            ++hits10;
        }
    }

    ++metrics->samples;
    metrics->positive_total += positive_count;
    metrics->top1_prime += targets[ranked[0]] > (nn_real)0 ? 1u : 0u;
    metrics->top5_any_prime += hits5 != 0 ? 1u : 0u;
    metrics->top10_any_prime += hits10 != 0 ? 1u : 0u;
    metrics->top1_wheel_survivor +=
        gap_survives_feature_wheel(p, indexed_gap(ranked[0])) ? 1u : 0u;
    metrics->positive_probability += positive_probability;
    metrics->nll += nll;
    metrics->mean_average_precision += average_precision / (double)positive_count;
    metrics->recall_at_5 += (double)hits5 / (double)positive_count;
    metrics->recall_at_10 += (double)hits10 / (double)positive_count;
    metrics->first_prime_rank += (double)first_rank + 1.0;
    metrics->top1_confidence += (double)probabilities[ranked[0]];
}

void finalize_interval_metrics(IntervalMetrics *metrics)
{
    double samples = metrics->samples == 0 ? 1.0 : (double)metrics->samples;

    metrics->positive_probability /= samples;
    metrics->nll /= samples;
    metrics->mean_average_precision /= samples;
    metrics->recall_at_5 /= samples;
    metrics->recall_at_10 /= samples;
    metrics->first_prime_rank /= samples;
    metrics->top1_confidence /= samples;
}

IntervalMetrics evaluate_interval_model(NeuralNet *net,
                                               const Sieve *sieve,
                                               const GapDataset *dataset,
                                               size_t max_states,
                                               size_t candidate_count,
                                               double gap_scale,
                                               double log_scale,
                                               const FeatureMask *mask,
                                               nn_real *candidate_inputs,
                                               nn_real *scores,
                                               nn_real *probabilities,
                                               nn_real *targets)
{
    IntervalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }
    if (candidate_count == 0 || candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        finalize_interval_metrics(&metrics);
        return metrics;
    }

    size_t ranked[INTERVAL_MAX_ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        size_t positive_count = build_interval_prime_targets(sieve, p, candidate_count, targets);
        if (positive_count == 0) {
            continue;
        }

        fill_interval_candidate_features(
            candidate_inputs, p, candidate_count, gap_scale, log_scale, mask);
        double profiled = prime_gap_profile_start();
        if (!nn_score_action_rows(net, candidate_inputs, candidate_count, scores, NULL)) {
            prime_gap_profile_stop(PRIME_GAP_PROFILE_FORWARD, profiled);
            break;
        }
        prime_gap_profile_stop(PRIME_GAP_PROFILE_FORWARD, profiled);

        profiled = prime_gap_profile_start();
        softmax_temperature(scores, probabilities, candidate_count, 1.0);
        rank_actions_count(scores, candidate_count, ranked);
        accumulate_interval_sample(&metrics,
                                   p,
                                   ranked,
                                   probabilities,
                                   targets,
                                   candidate_count,
                                   positive_count);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

IntervalMetrics evaluate_interval_model_cached(NeuralNet *net,
                                               const GapDataset *dataset,
                                               const IntervalTargetCache *cache,
                                               double gap_scale,
                                               double log_scale,
                                               const FeatureMask *mask,
                                               nn_real *candidate_inputs,
                                               nn_real *scores,
                                               nn_real *probabilities)
{
    IntervalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    if (cache == NULL || cache->candidate_count == 0 ||
        cache->candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        finalize_interval_metrics(&metrics);
        return metrics;
    }

    size_t ranked[INTERVAL_MAX_ACTION_COUNT];
    for (size_t sample = 0; sample < cache->sample_count; ++sample) {
        size_t index = eval_sample_index(dataset, sample, cache->sample_count);
        uint32_t p = dataset->prime[index];
        size_t positive_count = cache->positive_counts[sample];
        const nn_real *targets = &cache->targets[sample * cache->candidate_count];
        if (positive_count == 0) {
            continue;
        }

        fill_interval_candidate_features(candidate_inputs,
                                         p,
                                         cache->candidate_count,
                                         gap_scale,
                                         log_scale,
                                         mask);
        double profiled = prime_gap_profile_start();
        if (!nn_score_action_rows(net, candidate_inputs, cache->candidate_count, scores, NULL)) {
            prime_gap_profile_stop(PRIME_GAP_PROFILE_FORWARD, profiled);
            break;
        }
        prime_gap_profile_stop(PRIME_GAP_PROFILE_FORWARD, profiled);

        profiled = prime_gap_profile_start();
        softmax_temperature(scores, probabilities, cache->candidate_count, 1.0);
        rank_actions_count(scores, cache->candidate_count, ranked);
        accumulate_interval_sample(&metrics,
                                   p,
                                   ranked,
                                   probabilities,
                                   targets,
                                   cache->candidate_count,
                                   positive_count);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

double tune_temperature(NeuralNet *net,
                               const Sieve *sieve,
                               const GapDataset *dataset,
                               size_t max_states,
                               double log_scale,
                               const FeatureMask *mask,
                               nn_real *candidate_inputs,
                               nn_real *scores,
                               nn_real *probabilities)
{
    double best_temperature = 1.0;
    double best_nll = HUGE_VAL;
    size_t grid_count = sizeof(TEMPERATURE_GRID) / sizeof(TEMPERATURE_GRID[0]);

    for (size_t i = 0; i < grid_count; ++i) {
        double temperature = TEMPERATURE_GRID[i];
        EvalMetrics metrics = evaluate_model(net,
                                             sieve,
                                             dataset,
                                             max_states,
                                             log_scale,
                                             mask,
                                             temperature,
                                             0,
                                             0,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        if (metrics.nll < best_nll) {
            best_nll = metrics.nll;
            best_temperature = temperature;
        }
    }

    return best_temperature;
}

void print_metrics(const char *label, EvalMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
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
    printf("    samples=%zu temperature=%.2f top1=%.2f%% top5=%.2f%% top10=%.2f%% top1_prime=%.2f%%\n",
           metrics.samples,
           metrics.temperature,
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
    printf("    bins\n");
    for (size_t bin = 0; bin < CALIBRATION_BINS; ++bin) {
        if (metrics.calibration_count[bin] == 0) {
            continue;
        }

        printf("      %.1f-%.1f n=%zu conf=%.4f acc=%.4f\n",
               (double)bin / (double)CALIBRATION_BINS,
               (double)(bin + 1) / (double)CALIBRATION_BINS,
               metrics.calibration_count[bin],
               metrics.calibration_confidence[bin],
               metrics.calibration_correct[bin]);
    }

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

void print_interval_metrics(const char *label, IntervalMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    double average_positives = metrics.samples == 0
                                   ? 0.0
                                   : (double)metrics.positive_total / samples;

    printf("\n[%s]\n", label);
    printf("  interval capability\n");
    printf("    samples=%zu avg_primes_in_window=%.2f top1_prime=%.2f%% "
           "top5_any_prime=%.2f%% top10_any_prime=%.2f%%\n",
           metrics.samples,
           average_positives,
           100.0 * (double)metrics.top1_prime / samples,
           100.0 * (double)metrics.top5_any_prime / samples,
           100.0 * (double)metrics.top10_any_prime / samples);
    printf("    nll=%.4f p_any_prime=%.4f map=%.4f recall5=%.4f recall10=%.4f "
           "first_prime_rank=%.2f\n",
           metrics.nll,
           metrics.positive_probability,
           metrics.mean_average_precision,
           metrics.recall_at_5,
           metrics.recall_at_10,
           metrics.first_prime_rank);
    printf("    top1_conf=%.4f top1_survives_feature_wheel=%.2f%%\n",
           metrics.top1_confidence,
           100.0 * (double)metrics.top1_wheel_survivor / samples);
}

void print_calibration_comparison(const char *label,
                                         EvalMetrics supervised,
                                         EvalMetrics entropy,
                                         EvalMetrics strict_rl)
{
    printf("\n[%s]\n", label);
    printf("  model                 nll     p_true  top1_conf  ece     high_conf_acc\n");
    printf("  supervised            %.4f  %.4f  %.4f     %.4f  %.2f%%\n",
           supervised.nll,
           supervised.true_action_probability,
           supervised.top1_confidence,
           supervised.expected_calibration_error,
           supervised.high_confidence == 0
               ? 0.0
               : 100.0 * (double)supervised.high_confidence_correct /
                     (double)supervised.high_confidence);
    printf("  entropy-supervised    %.4f  %.4f  %.4f     %.4f  %.2f%%\n",
           entropy.nll,
           entropy.true_action_probability,
           entropy.top1_confidence,
           entropy.expected_calibration_error,
           entropy.high_confidence == 0
               ? 0.0
               : 100.0 * (double)entropy.high_confidence_correct /
                     (double)entropy.high_confidence);
    printf("  strict-rl             %.4f  %.4f  %.4f     %.4f  %.2f%%\n",
           strict_rl.nll,
           strict_rl.true_action_probability,
           strict_rl.top1_confidence,
           strict_rl.expected_calibration_error,
           strict_rl.high_confidence == 0
               ? 0.0
               : 100.0 * (double)strict_rl.high_confidence_correct /
                     (double)strict_rl.high_confidence);
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

void score_wheel_frequency_policy(const nn_real *base_probabilities,
                                         uint32_t p,
                                         size_t *ranked,
                                         nn_real *probabilities)
{
    nn_real scores[ACTION_COUNT];
    double sum = 0.0;
    size_t survivor_count = 0;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (!action_survives_feature_wheel(p, action)) {
            probabilities[action] = 0;
            scores[action] = 0;
            continue;
        }

        ++survivor_count;
        scores[action] = base_probabilities[action];
        sum += (double)scores[action];
    }

    if (sum <= 0.0) {
        nn_real uniform = survivor_count == 0
                              ? (nn_real)1 / (nn_real)ACTION_COUNT
                              : (nn_real)1 / (nn_real)survivor_count;
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = action_survives_feature_wheel(p, action) ? uniform : 0;
        }
    } else {
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = action_survives_feature_wheel(p, action)
                                        ? (nn_real)((double)scores[action] / sum)
                                        : (nn_real)0;
        }
    }

    rank_actions_wheel(scores, p, ranked);
}

EvalMetrics evaluate_wheel_frequency_policy(const Sieve *sieve,
                                                   const GapDataset *dataset,
                                                   const nn_real *base_probabilities,
                                                   size_t max_states,
                                                   int include_verifier,
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
        double profiled = prime_gap_profile_start();
        score_wheel_frequency_policy(
            base_probabilities, dataset->prime[index], ranked, probabilities);
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               dataset->prime[index],
                               dataset->gap[index],
                               ranked,
                               probabilities,
                               include_verifier);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    metrics.samples = samples;
    metrics.temperature = 1.0;
    finalize_eval_metrics(&metrics);
    return metrics;
}

void build_interval_frequency_policy(const Sieve *sieve,
                                            const GapDataset *dataset,
                                            size_t candidate_count,
                                            size_t *ranked,
                                            nn_real *probabilities)
{
    size_t counts[INTERVAL_MAX_ACTION_COUNT] = {0};
    nn_real scores[INTERVAL_MAX_ACTION_COUNT];
    double sum = 0.0;
    double profiled = prime_gap_profile_start();

    if (candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        candidate_count = INTERVAL_MAX_ACTION_COUNT;
    }

    for (size_t i = 0; i < dataset->count; ++i) {
        uint32_t p = dataset->prime[i];
        for (size_t action = 0; action < candidate_count; ++action) {
            uint32_t candidate = p + indexed_gap(action);
            if (is_prime_sieve(sieve, candidate)) {
                ++counts[action];
            }
        }
    }

    for (size_t action = 0; action < candidate_count; ++action) {
        scores[action] = (nn_real)((double)counts[action] + 1.0);
        sum += (double)scores[action];
    }

    for (size_t action = 0; action < candidate_count; ++action) {
        probabilities[action] = (nn_real)((double)scores[action] / sum);
    }

    rank_actions_count(scores, candidate_count, ranked);
    prime_gap_profile_stop(PRIME_GAP_PROFILE_TARGETS, profiled);
}

void rank_interval_wheel(const nn_real *scores,
                                uint32_t p,
                                size_t candidate_count,
                                size_t *ranked)
{
    size_t survivor_count = 0;
    size_t rejected_count = 0;
    size_t rejected[INTERVAL_MAX_ACTION_COUNT];

    for (size_t action = 0; action < candidate_count; ++action) {
        if (gap_survives_feature_wheel(p, indexed_gap(action))) {
            ranked[survivor_count++] = action;
        } else {
            rejected[rejected_count++] = action;
        }
    }

    for (size_t i = 0; i < survivor_count; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < survivor_count; ++j) {
            if (scores[ranked[j]] > scores[ranked[best]]) {
                best = j;
            }
        }

        size_t tmp = ranked[i];
        ranked[i] = ranked[best];
        ranked[best] = tmp;
    }

    for (size_t i = 0; i < rejected_count; ++i) {
        ranked[survivor_count + i] = rejected[i];
    }
}

void score_interval_wheel_frequency_policy(const nn_real *base_probabilities,
                                                  uint32_t p,
                                                  size_t candidate_count,
                                                  size_t *ranked,
                                                  nn_real *probabilities)
{
    nn_real scores[INTERVAL_MAX_ACTION_COUNT];
    double sum = 0.0;
    size_t survivor_count = 0;

    for (size_t action = 0; action < candidate_count; ++action) {
        if (!gap_survives_feature_wheel(p, indexed_gap(action))) {
            probabilities[action] = 0;
            scores[action] = 0;
            continue;
        }

        ++survivor_count;
        scores[action] = base_probabilities[action];
        sum += (double)scores[action];
    }

    if (sum <= 0.0) {
        nn_real uniform = survivor_count == 0
                              ? (nn_real)1 / (nn_real)candidate_count
                              : (nn_real)1 / (nn_real)survivor_count;
        for (size_t action = 0; action < candidate_count; ++action) {
            probabilities[action] =
                survivor_count == 0 || gap_survives_feature_wheel(p, indexed_gap(action))
                    ? uniform
                    : 0;
        }
    } else {
        for (size_t action = 0; action < candidate_count; ++action) {
            probabilities[action] = gap_survives_feature_wheel(p, indexed_gap(action))
                                        ? (nn_real)((double)scores[action] / sum)
                                        : (nn_real)0;
        }
    }

    rank_interval_wheel(scores, p, candidate_count, ranked);
}

IntervalMetrics evaluate_interval_static_policy(const Sieve *sieve,
                                                       const GapDataset *dataset,
                                                       const size_t *ranked,
                                                       const nn_real *probabilities,
                                                       size_t candidate_count,
                                                       size_t max_states,
                                                       nn_real *targets)
{
    IntervalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }
    if (candidate_count == 0 || candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        finalize_interval_metrics(&metrics);
        return metrics;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        size_t positive_count = build_interval_prime_targets(sieve, p, candidate_count, targets);
        if (positive_count == 0) {
            continue;
        }

        double profiled = prime_gap_profile_start();
        accumulate_interval_sample(
            &metrics, p, ranked, probabilities, targets, candidate_count, positive_count);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

IntervalMetrics evaluate_interval_static_policy_cached(const GapDataset *dataset,
                                                       const IntervalTargetCache *cache,
                                                       const size_t *ranked,
                                                       const nn_real *probabilities)
{
    IntervalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    if (cache == NULL || cache->candidate_count == 0 ||
        cache->candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        finalize_interval_metrics(&metrics);
        return metrics;
    }

    for (size_t sample = 0; sample < cache->sample_count; ++sample) {
        size_t index = eval_sample_index(dataset, sample, cache->sample_count);
        uint32_t p = dataset->prime[index];
        size_t positive_count = cache->positive_counts[sample];
        const nn_real *targets = &cache->targets[sample * cache->candidate_count];
        if (positive_count == 0) {
            continue;
        }

        double profiled = prime_gap_profile_start();
        accumulate_interval_sample(&metrics,
                                   p,
                                   ranked,
                                   probabilities,
                                   targets,
                                   cache->candidate_count,
                                   positive_count);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

IntervalMetrics evaluate_interval_wheel_frequency_policy(const Sieve *sieve,
                                                                const GapDataset *dataset,
                                                                const nn_real *base_probabilities,
                                                                size_t candidate_count,
                                                                size_t max_states,
                                                                nn_real *probabilities,
                                                                nn_real *targets)
{
    IntervalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }
    if (candidate_count == 0 || candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        finalize_interval_metrics(&metrics);
        return metrics;
    }

    size_t ranked[INTERVAL_MAX_ACTION_COUNT];
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        size_t positive_count = build_interval_prime_targets(sieve, p, candidate_count, targets);
        if (positive_count == 0) {
            continue;
        }

        double profiled = prime_gap_profile_start();
        score_interval_wheel_frequency_policy(
            base_probabilities, p, candidate_count, ranked, probabilities);
        accumulate_interval_sample(
            &metrics, p, ranked, probabilities, targets, candidate_count, positive_count);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

IntervalMetrics evaluate_interval_wheel_frequency_policy_cached(
    const GapDataset *dataset,
    const IntervalTargetCache *cache,
    const nn_real *base_probabilities,
    nn_real *probabilities)
{
    IntervalMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    if (cache == NULL || cache->candidate_count == 0 ||
        cache->candidate_count > INTERVAL_MAX_ACTION_COUNT) {
        finalize_interval_metrics(&metrics);
        return metrics;
    }

    size_t ranked[INTERVAL_MAX_ACTION_COUNT];
    for (size_t sample = 0; sample < cache->sample_count; ++sample) {
        size_t index = eval_sample_index(dataset, sample, cache->sample_count);
        uint32_t p = dataset->prime[index];
        size_t positive_count = cache->positive_counts[sample];
        const nn_real *targets = &cache->targets[sample * cache->candidate_count];
        if (positive_count == 0) {
            continue;
        }

        double profiled = prime_gap_profile_start();
        score_interval_wheel_frequency_policy(
            base_probabilities, p, cache->candidate_count, ranked, probabilities);
        accumulate_interval_sample(&metrics,
                                   p,
                                   ranked,
                                   probabilities,
                                   targets,
                                   cache->candidate_count,
                                   positive_count);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

void free_conditioned_policy(ConditionedPolicy *policy)
{
    free(policy->counts);
    free(policy->totals);
    memset(policy, 0, sizeof(*policy));
}

int build_residue_conditioned_policy(const GapDataset *dataset, ConditionedPolicy *policy)
{
    memset(policy, 0, sizeof(*policy));
    policy->counts = calloc((size_t)CONDITIONED_SIGNATURE_COUNT * ACTION_COUNT,
                            sizeof(*policy->counts));
    policy->totals = calloc(CONDITIONED_SIGNATURE_COUNT, sizeof(*policy->totals));
    if (policy->counts == NULL || policy->totals == NULL) {
        free_conditioned_policy(policy);
        return 0;
    }

    for (size_t i = 0; i < dataset->count; ++i) {
        size_t action = gap_to_action(dataset->gap[i]);
        size_t signature = candidate_signature(dataset->prime[i], action);
        ++policy->counts[signature * ACTION_COUNT + action];
        ++policy->totals[signature];
    }

    return 1;
}

void score_residue_conditioned_policy(const ConditionedPolicy *policy,
                                             uint32_t p,
                                             size_t *ranked,
                                             nn_real *probabilities)
{
    nn_real scores[ACTION_COUNT];
    double sum = 0.0;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        size_t signature = candidate_signature(p, action);
        size_t count = policy->counts[signature * ACTION_COUNT + action];
        size_t total = policy->totals[signature];
        scores[action] = (nn_real)(((double)count + 1.0) /
                                   ((double)total + (double)ACTION_COUNT));
        sum += (double)scores[action];
    }

    if (sum <= 0.0) {
        nn_real uniform = (nn_real)1 / (nn_real)ACTION_COUNT;
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = uniform;
        }
    } else {
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = (nn_real)((double)scores[action] / sum);
        }
    }

    rank_actions(scores, ranked);
}

EvalMetrics evaluate_residue_conditioned_policy(const ConditionedPolicy *policy,
                                                       const Sieve *sieve,
                                                       const GapDataset *dataset,
                                                       size_t max_states,
                                                       int include_verifier,
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
        double profiled = prime_gap_profile_start();
        score_residue_conditioned_policy(policy, dataset->prime[index], ranked, probabilities);
        accumulate_eval_sample(&metrics,
                               sieve,
                               dataset,
                               dataset->prime[index],
                               dataset->gap[index],
                               ranked,
                               probabilities,
                               include_verifier);
        prime_gap_profile_stop(PRIME_GAP_PROFILE_METRICS, profiled);
    }

    metrics.samples = samples;
    metrics.temperature = 1.0;
    finalize_eval_metrics(&metrics);
    return metrics;
}
