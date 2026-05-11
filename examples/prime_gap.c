#include "nn.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    G_MAX = 256,
    EXACT_ACTION_COUNT = G_MAX / 2,
    OVERFLOW_ACTION = EXACT_ACTION_COUNT,
    ACTION_COUNT = EXACT_ACTION_COUNT + 1,
    INTERVAL_MAX_GAP = 2048,
    INTERVAL_MAX_ACTION_COUNT = INTERVAL_MAX_GAP / 2,
    FEATURE_MOD_COUNT = 10,
    FEATURE_COUNT = 3 + FEATURE_MOD_COUNT,
    DEFAULT_TRAIN_LIMIT = 1000000,
    DEFAULT_TEST_LIMIT = 3000000,
    DEFAULT_SUPERVISED_EPOCHS = 2,
    DEFAULT_RL_EPOCHS = 1,
    DEFAULT_MAX_TRAIN_STATES = 30000,
    DEFAULT_MAX_TEST_STATES = 30000,
    CALIBRATION_BINS = 10,
    PRIME_BUCKET_COUNT = 4,
    RESIDUE_SIGNATURE_COUNT = 1u << FEATURE_MOD_COUNT,
    CONDITIONED_SIGNATURE_COUNT = RESIDUE_SIGNATURE_COUNT + 1,
};

static const uint32_t FEATURE_MODS[FEATURE_MOD_COUNT] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
static const uint32_t INTERVAL_SWEEP_GAPS[] = {512, 1024, 2048};
static const double TEMPERATURE_GRID[] = {0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00, 6.00, 8.00};
static const nn_real SUPERVISED_ENTROPY_WEIGHT = (nn_real)0.10;

enum {
    RESIDUE_FEATURE_NONE = 0,
    RESIDUE_FEATURE_DIVISIBILITY = 1,
    RESIDUE_FEATURE_RAW_NORMALIZED = 2,
};

typedef struct {
    const char *name;
    int use_log;
    int use_gap;
    int use_overflow;
    int residue_mode;
} FeatureMask;

static const FeatureMask FEATURE_MASK_FULL = {"full", 1, 1, 1, RESIDUE_FEATURE_DIVISIBILITY};
static const FeatureMask FEATURE_MASK_LOG_ONLY = {"log-only", 1, 0, 0, RESIDUE_FEATURE_NONE};
static const FeatureMask FEATURE_MASK_GAP_ONLY = {"gap-only", 0, 1, 1, RESIDUE_FEATURE_NONE};
static const FeatureMask FEATURE_MASK_RESIDUE_ONLY = {"residue-only", 0, 0, 1, RESIDUE_FEATURE_DIVISIBILITY};
static const FeatureMask FEATURE_MASK_RAW_RESIDUE = {"raw-residue", 1, 1, 0, RESIDUE_FEATURE_RAW_NORMALIZED};

typedef struct {
    uint8_t *composite;
    uint32_t *primes;
    size_t prime_count;
    uint32_t limit;
} Sieve;

typedef struct {
    uint32_t *prime;
    uint32_t *gap;
    size_t total_seen;
    size_t count;
    size_t capacity;
    size_t overflow_count;
} GapDataset;

typedef struct {
    size_t candidates;
    size_t mod_tests;
    size_t exact_tests;
} VerifierWork;

typedef struct {
    size_t candidates;
    size_t base_primes;
    size_t marks;
    size_t survivor_checks;
} BlockWork;

typedef struct {
    size_t *counts;
    size_t *totals;
} ConditionedPolicy;

typedef struct {
    size_t samples;
    size_t top1;
    size_t top5;
    size_t top10;
    size_t top1_prime;
    size_t correct_top1_confidence_count;
    size_t incorrect_top1_confidence_count;
    size_t high_confidence;
    size_t high_confidence_correct;
    size_t verifier_samples;
    size_t calibration_count[CALIBRATION_BINS];
    size_t bucket_samples[PRIME_BUCKET_COUNT];
    size_t bucket_top1[PRIME_BUCKET_COUNT];
    size_t bucket_top5[PRIME_BUCKET_COUNT];
    size_t bucket_top10[PRIME_BUCKET_COUNT];
    double nll;
    double true_action_probability;
    double mean_true_rank;
    double mean_reciprocal_rank;
    double top1_confidence;
    double correct_top1_confidence;
    double incorrect_top1_confidence;
    double expected_calibration_error;
    double calibration_confidence[CALIBRATION_BINS];
    double calibration_correct[CALIBRATION_BINS];
    double bucket_nll[PRIME_BUCKET_COUNT];
    double bucket_true_rank[PRIME_BUCKET_COUNT];
    double temperature;
    double mean_abs_gap_error;
    double direct_scan_exact_tests;
    double wheel_scan_mod_tests;
    double wheel_scan_exact_tests;
    double proposal_steps_top10;
    double assisted_wheel_mod_tests_top10;
    double assisted_wheel_exact_tests_top10;
    double block_scan_cost;
    double assisted_block_cost_top10;
} EvalMetrics;

typedef struct {
    size_t samples;
    size_t top1_prime;
    size_t top5_any_prime;
    size_t top10_any_prime;
    size_t top1_wheel_survivor;
    size_t positive_total;
    double positive_probability;
    double nll;
    double mean_average_precision;
    double recall_at_5;
    double recall_at_10;
    double first_prime_rank;
    double top1_confidence;
} IntervalMetrics;

static double now_seconds(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static uint32_t rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double rng_uniform(uint32_t *state)
{
    return (double)rng_next(state) / (double)UINT32_MAX;
}

static size_t rng_bounded(uint32_t *state, size_t bound)
{
    return bound == 0 ? 0 : (size_t)(rng_next(state) % (uint32_t)bound);
}

static size_t parse_size_arg(int argc, char **argv, int index, size_t fallback)
{
    if (index >= argc) {
        return fallback;
    }

    char *end = NULL;
    unsigned long value = strtoul(argv[index], &end, 10);
    return end != argv[index] && *end == '\0' ? (size_t)value : fallback;
}

static void free_sieve(Sieve *sieve)
{
    free(sieve->composite);
    free(sieve->primes);
    memset(sieve, 0, sizeof(*sieve));
}

static int build_sieve(uint32_t limit, Sieve *sieve)
{
    memset(sieve, 0, sizeof(*sieve));
    sieve->limit = limit;
    sieve->composite = calloc((size_t)limit + 1, sizeof(*sieve->composite));
    if (sieve->composite == NULL) {
        return 0;
    }

    if (limit >= 1) {
        sieve->composite[0] = 1;
        sieve->composite[1] = 1;
    }

    for (uint32_t p = 2; (uint64_t)p * (uint64_t)p <= limit; ++p) {
        if (sieve->composite[p]) {
            continue;
        }

        for (uint64_t multiple = (uint64_t)p * (uint64_t)p; multiple <= limit; multiple += p) {
            sieve->composite[multiple] = 1;
        }
    }

    for (uint32_t n = 2; n <= limit; ++n) {
        sieve->prime_count += sieve->composite[n] ? 0u : 1u;
    }

    sieve->primes = calloc(sieve->prime_count, sizeof(*sieve->primes));
    if (sieve->primes == NULL) {
        free_sieve(sieve);
        return 0;
    }

    size_t index = 0;
    for (uint32_t n = 2; n <= limit; ++n) {
        if (!sieve->composite[n]) {
            sieve->primes[index++] = n;
        }
    }

    return 1;
}

static int is_prime_sieve(const Sieve *sieve, uint32_t n)
{
    return n <= sieve->limit && n >= 2 && !sieve->composite[n];
}

static void free_gap_dataset(GapDataset *dataset)
{
    free(dataset->prime);
    free(dataset->gap);
    memset(dataset, 0, sizeof(*dataset));
}

static int append_gap(GapDataset *dataset, uint32_t prime, uint32_t gap)
{
    if (dataset->count == dataset->capacity) {
        size_t next_capacity = dataset->capacity == 0 ? 1024 : dataset->capacity * 2;
        uint32_t *next_prime = calloc(next_capacity, sizeof(*next_prime));
        uint32_t *next_gap = calloc(next_capacity, sizeof(*next_gap));
        if (next_prime == NULL || next_gap == NULL) {
            free(next_prime);
            free(next_gap);
            return 0;
        }

        if (dataset->count != 0) {
            memcpy(next_prime, dataset->prime, dataset->count * sizeof(*next_prime));
            memcpy(next_gap, dataset->gap, dataset->count * sizeof(*next_gap));
        }

        free(dataset->prime);
        free(dataset->gap);
        dataset->prime = next_prime;
        dataset->gap = next_gap;
        dataset->capacity = next_capacity;
    }

    dataset->prime[dataset->count] = prime;
    dataset->gap[dataset->count] = gap;
    ++dataset->count;
    return 1;
}

static int build_gap_datasets(const Sieve *sieve,
                              uint32_t train_limit,
                              uint32_t test_limit,
                              GapDataset *train,
                              GapDataset *test)
{
    memset(train, 0, sizeof(*train));
    memset(test, 0, sizeof(*test));

    for (size_t i = 0; i + 1 < sieve->prime_count; ++i) {
        uint32_t p = sieve->primes[i];
        uint32_t next = sieve->primes[i + 1];
        uint32_t gap = next - p;

        if (p < 3 || (gap & 1u) != 0) {
            continue;
        }

        GapDataset *target = NULL;
        if (p < train_limit) {
            target = train;
        } else if (p >= train_limit && next <= test_limit) {
            target = test;
        }

        if (target == NULL) {
            continue;
        }

        ++target->total_seen;
        if (gap > G_MAX) {
            ++target->overflow_count;
        }

        if (!append_gap(target, p, gap)) {
            free_gap_dataset(train);
            free_gap_dataset(test);
            return 0;
        }
    }

    return 1;
}

static int copy_gap_slice(const GapDataset *source, size_t start, size_t count, GapDataset *target)
{
    memset(target, 0, sizeof(*target));

    if (start > source->count) {
        return 0;
    }
    if (count > source->count - start) {
        count = source->count - start;
    }

    for (size_t i = 0; i < count; ++i) {
        uint32_t p = source->prime[start + i];
        uint32_t gap = source->gap[start + i];
        ++target->total_seen;
        if (gap > G_MAX) {
            ++target->overflow_count;
        }
        if (!append_gap(target, p, gap)) {
            free_gap_dataset(target);
            return 0;
        }
    }

    return 1;
}

static int split_train_calibration(const GapDataset *source,
                                   GapDataset *fit,
                                   GapDataset *calibration)
{
    size_t calibration_count = source->count / 5u;
    size_t fit_count;

    if (source->count < 2) {
        return 0;
    }
    if (calibration_count == 0) {
        calibration_count = 1;
    }
    fit_count = source->count - calibration_count;
    if (fit_count == 0) {
        return 0;
    }

    if (!copy_gap_slice(source, 0, fit_count, fit)) {
        return 0;
    }
    if (!copy_gap_slice(source, fit_count, calibration_count, calibration)) {
        free_gap_dataset(fit);
        return 0;
    }

    return 1;
}

static void shuffle_order(size_t *order, size_t count, uint32_t *rng)
{
    for (size_t i = count; i > 1; --i) {
        size_t j = rng_bounded(rng, i);
        size_t tmp = order[i - 1];
        order[i - 1] = order[j];
        order[j] = tmp;
    }
}

static size_t action_gap(size_t action)
{
    if (action == OVERFLOW_ACTION) {
        return G_MAX + 2;
    }

    return (action + 1) * 2;
}

static uint32_t indexed_gap(size_t action)
{
    return (uint32_t)((action + 1u) * 2u);
}

static size_t gap_to_action(uint32_t gap)
{
    if (gap > G_MAX) {
        return OVERFLOW_ACTION;
    }

    return (size_t)(gap / 2u - 1u);
}

static int action_is_overflow(size_t action)
{
    return action == OVERFLOW_ACTION;
}

static double gap_error_for_action(size_t action, uint32_t true_gap)
{
    if (action_is_overflow(action) && true_gap > G_MAX) {
        return 0.0;
    }

    return fabs((double)action_gap(action) - (double)true_gap);
}

static size_t candidate_gap_signature(uint32_t p, uint32_t gap)
{
    size_t signature = 0;
    uint32_t candidate = p + gap;
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate > mod && candidate % mod == 0) {
            signature |= (size_t)1u << i;
        }
    }

    return signature;
}

static size_t candidate_signature(uint32_t p, size_t action)
{
    if (action_is_overflow(action)) {
        return RESIDUE_SIGNATURE_COUNT;
    }

    return candidate_gap_signature(p, (uint32_t)action_gap(action));
}

static int gap_survives_feature_wheel(uint32_t p, uint32_t gap)
{
    return candidate_gap_signature(p, gap) == 0;
}

static int action_survives_feature_wheel(uint32_t p, size_t action)
{
    return action_is_overflow(action) || candidate_signature(p, action) == 0;
}

static void fill_gap_features(nn_real *row,
                              uint32_t p,
                              uint32_t gap,
                              int overflow,
                              double gap_scale,
                              double log_scale,
                              const FeatureMask *mask)
{
    if (mask == NULL) {
        mask = &FEATURE_MASK_FULL;
    }

    double log_p = log((double)p) / log_scale;
    row[0] = mask->use_log ? (nn_real)log_p : (nn_real)0;
    row[1] = mask->use_gap ? (nn_real)((double)gap / gap_scale) : (nn_real)0;
    row[2] = mask->use_overflow && overflow ? (nn_real)1 : (nn_real)0;
    if (overflow) {
        for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
            row[3 + i] = 0;
        }
        return;
    }

    uint32_t candidate = p + gap;
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        uint32_t residue = candidate % mod;

        if (mask->residue_mode == RESIDUE_FEATURE_DIVISIBILITY) {
            row[3 + i] = candidate > mod && residue == 0 ? (nn_real)1 : (nn_real)0;
        } else if (mask->residue_mode == RESIDUE_FEATURE_RAW_NORMALIZED) {
            row[3 + i] = (nn_real)((double)residue / (double)mod);
        } else {
            row[3 + i] = 0;
        }
    }
}

static void fill_action_features(nn_real *row,
                                 uint32_t p,
                                 size_t action,
                                 double log_scale,
                                 const FeatureMask *mask)
{
    fill_gap_features(row,
                      p,
                      (uint32_t)action_gap(action),
                      action_is_overflow(action),
                      (double)G_MAX,
                      log_scale,
                      mask);
}

static void fill_candidate_features_masked(nn_real *inputs,
                                           uint32_t p,
                                           double log_scale,
                                           const FeatureMask *mask)
{

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        fill_action_features(&inputs[action * FEATURE_COUNT], p, action, log_scale, mask);
    }
}

static size_t fill_wheel_survivor_features(nn_real *inputs,
                                           uint32_t p,
                                           size_t true_action,
                                           double log_scale,
                                           const FeatureMask *mask,
                                           size_t *target_index)
{
    size_t count = 0;
    *target_index = ACTION_COUNT;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (!action_survives_feature_wheel(p, action)) {
            continue;
        }

        if (action == true_action) {
            *target_index = count;
        }
        fill_action_features(&inputs[count * FEATURE_COUNT], p, action, log_scale, mask);
        ++count;
    }

    return count;
}

#if defined(PRIME_GAP_NO_MAIN)
static void fill_candidate_features(nn_real *inputs, uint32_t p, double log_scale)
{
    fill_candidate_features_masked(inputs, p, log_scale, &FEATURE_MASK_FULL);
}
#endif

static void rank_actions(const nn_real *scores, size_t *ranked)
{
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        ranked[i] = i;
    }

    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < ACTION_COUNT; ++j) {
            if (scores[ranked[j]] > scores[ranked[best]]) {
                best = j;
            }
        }

        size_t tmp = ranked[i];
        ranked[i] = ranked[best];
        ranked[best] = tmp;
    }
}

static void rank_actions_count(const nn_real *scores, size_t count, size_t *ranked)
{
    for (size_t i = 0; i < count; ++i) {
        ranked[i] = i;
    }

    for (size_t i = 0; i < count; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < count; ++j) {
            if (scores[ranked[j]] > scores[ranked[best]]) {
                best = j;
            }
        }

        size_t tmp = ranked[i];
        ranked[i] = ranked[best];
        ranked[best] = tmp;
    }
}

static void rank_actions_wheel(const nn_real *scores, uint32_t p, size_t *ranked)
{
    size_t survivor_count = 0;
    size_t rejected_count = 0;
    size_t rejected[ACTION_COUNT];

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (action_survives_feature_wheel(p, action)) {
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

static void softmax_temperature(const nn_real *scores,
                                nn_real *probabilities,
                                size_t count,
                                double temperature)
{
    if (temperature <= 0.0) {
        temperature = 1.0;
    }

    nn_real max_score = (nn_real)((double)scores[0] / temperature);
    for (size_t i = 1; i < count; ++i) {
        nn_real scaled = (nn_real)((double)scores[i] / temperature);
        if (scaled > max_score) {
            max_score = scaled;
        }
    }

    nn_real denominator = 0;
    for (size_t i = 0; i < count; ++i) {
        probabilities[i] = (nn_real)exp((double)scores[i] / temperature - (double)max_score);
        denominator += probabilities[i];
    }

    if (denominator <= (nn_real)0) {
        nn_real uniform = (nn_real)1 / (nn_real)count;
        for (size_t i = 0; i < count; ++i) {
            probabilities[i] = uniform;
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        probabilities[i] /= denominator;
    }
}

static void softmax_temperature_wheel(const nn_real *scores,
                                      nn_real *probabilities,
                                      uint32_t p,
                                      double temperature)
{
    int have_score = 0;
    size_t survivor_count = 0;
    nn_real max_score = 0;

    if (temperature <= 0.0) {
        temperature = 1.0;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] = 0;
        if (!action_survives_feature_wheel(p, action)) {
            continue;
        }

        ++survivor_count;
        nn_real scaled = (nn_real)((double)scores[action] / temperature);
        if (!have_score || scaled > max_score) {
            max_score = scaled;
            have_score = 1;
        }
    }

    if (!have_score) {
        nn_real uniform = (nn_real)1 / (nn_real)ACTION_COUNT;
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = uniform;
        }
        return;
    }

    nn_real denominator = 0;
    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (!action_survives_feature_wheel(p, action)) {
            continue;
        }

        probabilities[action] =
            (nn_real)exp((double)scores[action] / temperature - (double)max_score);
        denominator += probabilities[action];
    }

    if (denominator <= (nn_real)0) {
        nn_real uniform = survivor_count == 0
                              ? (nn_real)1 / (nn_real)ACTION_COUNT
                              : (nn_real)1 / (nn_real)survivor_count;
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = action_survives_feature_wheel(p, action) ? uniform : 0;
        }
        return;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] /= denominator;
    }
}

static size_t true_rank(const size_t *ranked, size_t true_action)
{
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        if (ranked[i] == true_action) {
            return i;
        }
    }

    return ACTION_COUNT;
}

static size_t proposal_steps_with_topk(const size_t *ranked, size_t top_k, size_t true_action)
{
    uint8_t tried[ACTION_COUNT] = {0};
    size_t steps = 0;

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t action = ranked[i];
        if (tried[action]) {
            continue;
        }

        tried[action] = 1;
        ++steps;
        if (action == true_action) {
            return steps;
        }
    }

    for (size_t action = 0; action <= true_action; ++action) {
        if (tried[action]) {
            continue;
        }

        ++steps;
        if (action == true_action) {
            return steps;
        }
    }

    return steps;
}

static int wheel_probe_gap(const Sieve *sieve, uint32_t p, uint32_t gap, VerifierWork *work)
{
    uint32_t candidate = p + gap;

    ++work->candidates;
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate <= mod) {
            continue;
        }

        ++work->mod_tests;
        if (candidate % mod == 0) {
            return 0;
        }
    }

    ++work->exact_tests;
    return is_prime_sieve(sieve, candidate);
}

static int wheel_probe_action_cached(const Sieve *sieve,
                                     uint32_t p,
                                     size_t action,
                                     uint8_t *tested,
                                     uint8_t *prime_result,
                                     VerifierWork *work)
{
    if (action_is_overflow(action)) {
        return 0;
    }

    if (!tested[action]) {
        tested[action] = 1;
        prime_result[action] =
            (uint8_t)wheel_probe_gap(sieve, p, (uint32_t)action_gap(action), work);
    }

    return prime_result[action] != 0;
}

static int wheel_scan_to_gap_cached(const Sieve *sieve,
                                    uint32_t p,
                                    uint32_t limit_gap,
                                    uint8_t *tested,
                                    uint8_t *prime_result,
                                    VerifierWork *work)
{
    for (uint32_t gap = 2; gap <= limit_gap; gap += 2) {
        int is_prime = 0;
        if (gap <= G_MAX) {
            is_prime = wheel_probe_action_cached(
                sieve, p, gap_to_action(gap), tested, prime_result, work);
        } else {
            is_prime = wheel_probe_gap(sieve, p, gap, work);
        }

        if (is_prime) {
            return 1;
        }
    }

    return 0;
}

static VerifierWork direct_scan_work(uint32_t true_gap)
{
    VerifierWork work;
    work.candidates = true_gap / 2u;
    work.mod_tests = 0;
    work.exact_tests = true_gap / 2u;
    return work;
}

static VerifierWork wheel_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap)
{
    uint8_t tested[EXACT_ACTION_COUNT] = {0};
    uint8_t prime_result[EXACT_ACTION_COUNT] = {0};
    VerifierWork work = {0, 0, 0};

    wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
    return work;
}

static VerifierWork assisted_wheel_work(const Sieve *sieve,
                                        uint32_t p,
                                        uint32_t true_gap,
                                        const size_t *ranked,
                                        size_t top_k)
{
    uint8_t tested[EXACT_ACTION_COUNT] = {0};
    uint8_t prime_result[EXACT_ACTION_COUNT] = {0};
    VerifierWork work = {0, 0, 0};

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        if (action_is_overflow(proposed)) {
            wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
            return work;
        }

        if (!wheel_probe_action_cached(sieve, p, proposed, tested, prime_result, &work)) {
            continue;
        }

        wheel_scan_to_gap_cached(
            sieve, p, (uint32_t)action_gap(proposed), tested, prime_result, &work);
        return work;
    }

    wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
    return work;
}

static void add_block_work(BlockWork *total, BlockWork part)
{
    total->candidates += part.candidates;
    total->base_primes += part.base_primes;
    total->marks += part.marks;
    total->survivor_checks += part.survivor_checks;
}

static double block_work_cost(BlockWork work)
{
    return (double)work.base_primes + (double)work.marks + (double)work.survivor_checks;
}

static uint32_t segmented_first_prime_gap(const Sieve *sieve,
                                          uint32_t p,
                                          uint32_t limit_gap,
                                          BlockWork *work)
{
    uint32_t candidate_count = limit_gap / 2u;
    uint8_t local_composite[EXACT_ACTION_COUNT];
    uint8_t *composite = local_composite;
    uint32_t start_number = p + 2u;
    uint32_t end_number = p + limit_gap;
    uint32_t found_gap = 0;

    memset(work, 0, sizeof(*work));
    work->candidates = candidate_count;

    if (candidate_count == 0) {
        return 0;
    }

    if (candidate_count <= EXACT_ACTION_COUNT) {
        memset(local_composite, 0, candidate_count * sizeof(*local_composite));
    } else {
        composite = calloc(candidate_count, sizeof(*composite));
        if (composite == NULL) {
            return 0;
        }
    }

    for (size_t i = 0; i < sieve->prime_count; ++i) {
        uint32_t prime = sieve->primes[i];
        uint64_t prime_squared = (uint64_t)prime * (uint64_t)prime;
        if (prime == 2) {
            continue;
        }
        if (prime_squared > end_number) {
            break;
        }

        ++work->base_primes;

        uint64_t first = ((uint64_t)start_number + prime - 1u) / prime * prime;
        if (first < prime_squared) {
            first = prime_squared;
        }
        if ((first & 1u) == 0) {
            first += prime;
        }

        for (uint64_t value = first; value <= end_number; value += (uint64_t)2u * prime) {
            uint32_t gap = (uint32_t)(value - p);
            if (gap < 2 || (gap & 1u) != 0) {
                continue;
            }

            uint32_t index = gap / 2u - 1u;
            if (index < candidate_count && !composite[index]) {
                composite[index] = 1;
                ++work->marks;
            }
        }
    }

    for (uint32_t index = 0; index < candidate_count; ++index) {
        ++work->survivor_checks;
        if (!composite[index]) {
            found_gap = (index + 1u) * 2u;
            break;
        }
    }

    if (composite != local_composite) {
        free(composite);
    }

    return found_gap;
}

static BlockWork block_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap)
{
    BlockWork work;
    segmented_first_prime_gap(sieve, p, true_gap, &work);
    return work;
}

static BlockWork assisted_block_work(const Sieve *sieve,
                                     uint32_t p,
                                     uint32_t true_gap,
                                     const size_t *ranked,
                                     size_t top_k)
{
    BlockWork total = {0, 0, 0, 0};

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        BlockWork part;
        uint32_t found_gap;

        if (action_is_overflow(proposed)) {
            add_block_work(&total, block_scan_work(sieve, p, true_gap));
            return total;
        }

        found_gap =
            segmented_first_prime_gap(sieve, p, (uint32_t)action_gap(proposed), &part);
        add_block_work(&total, part);

        if (found_gap != 0) {
            return total;
        }
    }

    add_block_work(&total, block_scan_work(sieve, p, true_gap));
    return total;
}

static size_t sample_action(const nn_real *probabilities, uint32_t *rng)
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

static double safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

static size_t eval_sample_index(const GapDataset *dataset, size_t sample, size_t samples)
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

static EvalMetrics evaluate_model(NeuralNet *net,
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
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];

        fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);
        nn_forward_batch(net, candidate_inputs, scores, ACTION_COUNT);
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
    }

    metrics.samples = samples;
    metrics.temperature = temperature <= 0.0 ? 1.0 : temperature;
    finalize_eval_metrics(&metrics);
    return metrics;
}

static EvalMetrics evaluate_static_policy(const Sieve *sieve,
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
    metrics.temperature = 1.0;
    finalize_eval_metrics(&metrics);
    return metrics;
}

static void fill_interval_candidate_features(nn_real *inputs,
                                             uint32_t p,
                                             size_t candidate_count,
                                             double gap_scale,
                                             double log_scale,
                                             const FeatureMask *mask)
{
    for (size_t action = 0; action < candidate_count; ++action) {
        fill_gap_features(&inputs[action * FEATURE_COUNT],
                          p,
                          indexed_gap(action),
                          0,
                          gap_scale,
                          log_scale,
                          mask);
    }
}

static size_t build_interval_prime_targets(const Sieve *sieve,
                                           uint32_t p,
                                           size_t candidate_count,
                                           nn_real *targets)
{
    size_t positive_count = 0;

    for (size_t action = 0; action < candidate_count; ++action) {
        uint32_t candidate = p + indexed_gap(action);
        int positive = is_prime_sieve(sieve, candidate);
        targets[action] = positive ? (nn_real)1 : (nn_real)0;
        positive_count += positive ? 1u : 0u;
    }

    if (positive_count == 0) {
        return 0;
    }

    nn_real target = (nn_real)1 / (nn_real)positive_count;
    for (size_t action = 0; action < candidate_count; ++action) {
        if (targets[action] > (nn_real)0) {
            targets[action] = target;
        }
    }

    return positive_count;
}

static void accumulate_interval_sample(IntervalMetrics *metrics,
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

static void finalize_interval_metrics(IntervalMetrics *metrics)
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

static IntervalMetrics evaluate_interval_model(NeuralNet *net,
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
        nn_forward_batch(net, candidate_inputs, scores, candidate_count);
        softmax_temperature(scores, probabilities, candidate_count, 1.0);
        rank_actions_count(scores, candidate_count, ranked);
        accumulate_interval_sample(&metrics,
                                   p,
                                   ranked,
                                   probabilities,
                                   targets,
                                   candidate_count,
                                   positive_count);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

static double tune_temperature(NeuralNet *net,
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

static void print_metrics(const char *label, EvalMetrics metrics)
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

static void print_interval_metrics(const char *label, IntervalMetrics metrics)
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

static void print_calibration_comparison(const char *label,
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

static void build_frequency_policy(const GapDataset *dataset,
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

static void score_wheel_frequency_policy(const nn_real *base_probabilities,
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

static EvalMetrics evaluate_wheel_frequency_policy(const Sieve *sieve,
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
    }

    metrics.samples = samples;
    metrics.temperature = 1.0;
    finalize_eval_metrics(&metrics);
    return metrics;
}

static void build_interval_frequency_policy(const Sieve *sieve,
                                            const GapDataset *dataset,
                                            size_t candidate_count,
                                            size_t *ranked,
                                            nn_real *probabilities)
{
    size_t counts[INTERVAL_MAX_ACTION_COUNT] = {0};
    nn_real scores[INTERVAL_MAX_ACTION_COUNT];
    double sum = 0.0;

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
}

static void rank_interval_wheel(const nn_real *scores,
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

static void score_interval_wheel_frequency_policy(const nn_real *base_probabilities,
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

static IntervalMetrics evaluate_interval_static_policy(const Sieve *sieve,
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

        accumulate_interval_sample(
            &metrics, p, ranked, probabilities, targets, candidate_count, positive_count);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

static IntervalMetrics evaluate_interval_wheel_frequency_policy(const Sieve *sieve,
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

        score_interval_wheel_frequency_policy(
            base_probabilities, p, candidate_count, ranked, probabilities);
        accumulate_interval_sample(
            &metrics, p, ranked, probabilities, targets, candidate_count, positive_count);
    }

    finalize_interval_metrics(&metrics);
    return metrics;
}

static void free_conditioned_policy(ConditionedPolicy *policy)
{
    free(policy->counts);
    free(policy->totals);
    memset(policy, 0, sizeof(*policy));
}

static int build_residue_conditioned_policy(const GapDataset *dataset, ConditionedPolicy *policy)
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

static void score_residue_conditioned_policy(const ConditionedPolicy *policy,
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

static EvalMetrics evaluate_residue_conditioned_policy(const ConditionedPolicy *policy,
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
        score_residue_conditioned_policy(policy, dataset->prime[index], ranked, probabilities);
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
    metrics.temperature = 1.0;
    finalize_eval_metrics(&metrics);
    return metrics;
}

static int train_supervised(NeuralNet *net,
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

static int train_interval_supervised(NeuralNet *net,
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
            nn_real step_loss = nn_train_softmax_scorer_targets(net,
                                                                candidate_inputs,
                                                                candidate_count,
                                                                targets,
                                                                learning_rate);
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

static int train_strict_verifier_rl(NeuralNet *net,
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
            nn_forward_batch(net, candidate_inputs, scores, ACTION_COUNT);
            nn_softmax(scores, probabilities, ACTION_COUNT);

            size_t selected = sample_action(probabilities, rng);
            double reward = selected == target ? 1.0 : 0.0;
            double advantage = reward - baseline;

            baseline += 0.01 * (reward - baseline);
            exact_hits += reward > 0.0 ? 1u : 0u;

            nn_real step_loss = nn_train_softmax_scorer_cached(net,
                                                               ACTION_COUNT,
                                                               selected,
                                                               (nn_real)advantage,
                                                               learning_rate);
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

static int run_supervised_ablation(const char *label,
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
                                   uint32_t *rng)
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
        print_metrics(label,
                      evaluate_model(net,
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
                                     probabilities));
    }

    nn_free(net);
    return ok;
}

#ifndef PRIME_GAP_NO_MAIN
int main(int argc, char **argv)
{
    size_t train_limit_arg = parse_size_arg(argc, argv, 1, DEFAULT_TRAIN_LIMIT);
    size_t test_limit_arg = parse_size_arg(argc, argv, 2, DEFAULT_TEST_LIMIT);
    size_t supervised_epochs = parse_size_arg(argc, argv, 3, DEFAULT_SUPERVISED_EPOCHS);
    size_t rl_epochs = parse_size_arg(argc, argv, 4, DEFAULT_RL_EPOCHS);
    size_t max_train_states = parse_size_arg(argc, argv, 5, DEFAULT_MAX_TRAIN_STATES);
    size_t max_test_states = parse_size_arg(argc, argv, 6, DEFAULT_MAX_TEST_STATES);
    Sieve sieve = {0, 0, 0, 0};
    GapDataset train = {0, 0, 0, 0, 0, 0};
    GapDataset fit = {0, 0, 0, 0, 0, 0};
    GapDataset calibration = {0, 0, 0, 0, 0, 0};
    GapDataset test = {0, 0, 0, 0, 0, 0};
    ConditionedPolicy residue_policy = {0, 0};
    uint32_t rng = 20240511u;
    EvalMetrics supervised_metrics;
    EvalMetrics supervised_temperature_metrics;
    EvalMetrics entropy_metrics;
    EvalMetrics entropy_temperature_metrics;
    EvalMetrics strict_rl_metrics;
    EvalMetrics strict_rl_temperature_metrics;
    double supervised_temperature = 1.0;
    double entropy_temperature = 1.0;
    double strict_rl_temperature = 1.0;
    int have_strict_rl_metrics = 0;
    int ok = 1;

    if (test_limit_arg <= train_limit_arg || train_limit_arg < 16 ||
        test_limit_arg > (size_t)UINT32_MAX - (size_t)G_MAX ||
        train_limit_arg > (size_t)UINT32_MAX) {
        fputs("usage: prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] "
              "[max_train_states] [max_test_states]\n",
              stderr);
        return EXIT_FAILURE;
    }

    uint32_t train_limit = (uint32_t)train_limit_arg;
    uint32_t test_limit = (uint32_t)test_limit_arg;
    uint32_t sieve_limit = test_limit + (uint32_t)INTERVAL_MAX_GAP;

    printf("prime-gap scorer train_limit=%u test_limit=%u g_max=%d actions=%d features=%d\n",
           train_limit,
           test_limit,
           G_MAX,
           ACTION_COUNT,
           FEATURE_COUNT);

    double started = now_seconds();
    if (!build_sieve(sieve_limit, &sieve)) {
        fputs("failed to build sieve\n", stderr);
        return EXIT_FAILURE;
    }
    printf("sieve limit=%u primes=%zu time=%.3fs\n",
           sieve_limit,
           sieve.prime_count,
           now_seconds() - started);

    if (!build_gap_datasets(&sieve, train_limit, test_limit, &train, &test)) {
        fputs("failed to build gap datasets\n", stderr);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    if (!split_train_calibration(&train, &fit, &calibration)) {
        fputs("failed to split fit/calibration datasets\n", stderr);
        free_gap_dataset(&train);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    printf("dataset raw_train=%zu fit=%zu calibration=%zu test=%zu "
           "exact_action_coverage train=%.2f%% test=%.2f%% "
           "overflow_actions train=%zu calibration=%zu test=%zu\n",
           train.count,
           fit.count,
           calibration.count,
           test.count,
           train.total_seen == 0
               ? 0.0
               : 100.0 * (double)(train.total_seen - train.overflow_count) /
                     (double)train.total_seen,
           test.total_seen == 0
               ? 0.0
               : 100.0 * (double)(test.total_seen - test.overflow_count) /
                     (double)test.total_seen,
           train.overflow_count,
           calibration.overflow_count,
           test.overflow_count);

    if (fit.count == 0 || calibration.count == 0 || test.count == 0) {
        fputs("empty fit, calibration, or test split\n", stderr);
        free_gap_dataset(&train);
        free_gap_dataset(&fit);
        free_gap_dataset(&calibration);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    const size_t layers[] = {FEATURE_COUNT, 32, 16, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 42;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    const size_t linear_layers[] = {FEATURE_COUNT, 1};
    NNConfig linear_config = config;
    linear_config.layer_sizes = linear_layers;
    linear_config.layer_count = sizeof(linear_layers) / sizeof(linear_layers[0]);

    NeuralNet *net = nn_create_config(&config);
    NeuralNet *entropy_net = NULL;
    NeuralNet *wheel_net = NULL;
    NeuralNet *interval_net = NULL;
    NeuralNet *interval_linear_net = NULL;
    NeuralNet *interval_raw_net = NULL;
    NeuralNet *interval_raw_linear_net = NULL;
    size_t max_candidate_rows = ACTION_COUNT > INTERVAL_MAX_ACTION_COUNT
                                    ? ACTION_COUNT
                                    : INTERVAL_MAX_ACTION_COUNT;
    nn_real *candidate_inputs =
        calloc(max_candidate_rows * FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(max_candidate_rows, sizeof(*scores));
    nn_real *probabilities = calloc(max_candidate_rows, sizeof(*probabilities));
    nn_real *target_distribution = calloc(INTERVAL_MAX_ACTION_COUNT, sizeof(*target_distribution));
    size_t *order = calloc(fit.count, sizeof(*order));
    if (net == NULL || candidate_inputs == NULL || scores == NULL || probabilities == NULL ||
        target_distribution == NULL || order == NULL) {
        fputs("failed to allocate model buffers\n", stderr);
        ok = 0;
        goto cleanup;
    }

    for (size_t i = 0; i < fit.count; ++i) {
        order[i] = i;
    }

    double log_scale = log((double)test_limit + (double)G_MAX);
    size_t frequency_ranked[ACTION_COUNT];
    nn_real frequency_probabilities[ACTION_COUNT];

    printf("\n=== Baselines ===\n");
    build_frequency_policy(&fit, frequency_ranked, frequency_probabilities);
    print_metrics("baseline empirical-gap",
                  evaluate_static_policy(&sieve,
                                         &test,
                                         frequency_ranked,
                                         frequency_probabilities,
                                         max_test_states,
                                         1));
    print_metrics("baseline wheel-frequency",
                  evaluate_wheel_frequency_policy(&sieve,
                                                  &test,
                                                  frequency_probabilities,
                                                  max_test_states,
                                                  0,
                                                  probabilities));

    if (!build_residue_conditioned_policy(&fit, &residue_policy)) {
        fputs("failed to build residue-conditioned baseline\n", stderr);
        ok = 0;
        goto cleanup;
    }
    print_metrics("baseline residue-conditioned",
                  evaluate_residue_conditioned_policy(&residue_policy,
                                                      &sieve,
                                                      &test,
                                                      max_test_states,
                                                      1,
                                                      probabilities));

    print_metrics("model before",
                  evaluate_model(net,
                                 &sieve,
                                 &test,
                                 max_test_states,
                                 log_scale,
                                 &FEATURE_MASK_FULL,
                                 1.0,
                                 0,
                                 1,
                                 candidate_inputs,
                                 scores,
                                 probabilities));

    printf("\n=== Feature Ablations ===\n");
    printf("Supervised-only runs; verifier diagnostics skipped to keep these capability-focused.\n");
    ok = run_supervised_ablation("baseline linear-full",
                                 &FEATURE_MASK_FULL,
                                 &linear_config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    ok = run_supervised_ablation("ablation log-only",
                                 &FEATURE_MASK_LOG_ONLY,
                                 &config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    ok = run_supervised_ablation("ablation gap-only",
                                 &FEATURE_MASK_GAP_ONLY,
                                 &config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    ok = run_supervised_ablation("ablation residue-only",
                                 &FEATURE_MASK_RESIDUE_ONLY,
                                 &config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    printf("\n=== Full Model Training ===\n");
    ok = train_supervised(net,
                          "model supervised",
                          &fit,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          (nn_real)0,
                          0,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    supervised_metrics = evaluate_model(net,
                                        &sieve,
                                        &test,
                                        max_test_states,
                                        log_scale,
                                        &FEATURE_MASK_FULL,
                                        1.0,
                                        0,
                                        1,
                                        candidate_inputs,
                                        scores,
                                        probabilities);
    print_metrics("model supervised", supervised_metrics);

    supervised_temperature = tune_temperature(net,
                                              &sieve,
                                              &calibration,
                                              max_train_states,
                                              log_scale,
                                              &FEATURE_MASK_FULL,
                                              candidate_inputs,
                                              scores,
                                              probabilities);
    supervised_temperature_metrics = evaluate_model(net,
                                                    &sieve,
                                                    &test,
                                                    max_test_states,
                                                    log_scale,
                                                    &FEATURE_MASK_FULL,
                                                    supervised_temperature,
                                                    0,
                                                    0,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
    print_metrics("model supervised temp-scaled", supervised_temperature_metrics);

    printf("\n=== Wheel-Survivor Task ===\n");
    printf("Ranks only candidates that survive the small-prime feature wheel; verifier diagnostics skipped.\n");
    print_metrics("model supervised wheel-survivor eval",
                  evaluate_model(net,
                                 &sieve,
                                 &test,
                                 max_test_states,
                                 log_scale,
                                 &FEATURE_MASK_FULL,
                                 1.0,
                                 1,
                                 0,
                                 candidate_inputs,
                                 scores,
                                 probabilities));

    wheel_net = nn_create_config(&config);
    if (wheel_net == NULL) {
        fputs("failed to allocate wheel-survivor model\n", stderr);
        ok = 0;
        goto cleanup;
    }

    ok = train_supervised(wheel_net,
                          "model wheel-survivor-supervised",
                          &fit,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          (nn_real)0,
                          1,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    print_metrics("model wheel-survivor-supervised",
                  evaluate_model(wheel_net,
                                 &sieve,
                                 &test,
                                 max_test_states,
                                 log_scale,
                                 &FEATURE_MASK_FULL,
                                 1.0,
                                 1,
                                 0,
                                 candidate_inputs,
                                 scores,
                                 probabilities));

    printf("\n=== Interval Prime Ranking Task ===\n");
    printf("Target is a distribution over every prime p+g in the 2..%d gap window, not only the next gap.\n",
           G_MAX);
    size_t interval_frequency_ranked[INTERVAL_MAX_ACTION_COUNT];
    nn_real interval_frequency_probabilities[INTERVAL_MAX_ACTION_COUNT];
    build_interval_frequency_policy(&sieve,
                                    &fit,
                                    EXACT_ACTION_COUNT,
                                    interval_frequency_ranked,
                                    interval_frequency_probabilities);
    print_interval_metrics("baseline interval-frequency",
                           evaluate_interval_static_policy(&sieve,
                                                           &test,
                                                           interval_frequency_ranked,
                                                           interval_frequency_probabilities,
                                                           EXACT_ACTION_COUNT,
                                                           max_test_states,
                                                           target_distribution));
    print_interval_metrics("baseline interval-wheel-frequency",
                           evaluate_interval_wheel_frequency_policy(&sieve,
                                                                    &test,
                                                                    interval_frequency_probabilities,
                                                                    EXACT_ACTION_COUNT,
                                                                    max_test_states,
                                                                    probabilities,
                                                                    target_distribution));

    interval_linear_net = nn_create_config(&linear_config);
    if (interval_linear_net == NULL) {
        fputs("failed to allocate interval linear model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_linear_net,
                                   "model interval-linear",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_FULL,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    print_interval_metrics("model interval-linear",
                           evaluate_interval_model(interval_linear_net,
                                                   &sieve,
                                                   &test,
                                                   max_test_states,
                                                   EXACT_ACTION_COUNT,
                                                   (double)G_MAX,
                                                   log_scale,
                                                   &FEATURE_MASK_FULL,
                                                   candidate_inputs,
                                                   scores,
                                                   probabilities,
                                                   target_distribution));

    interval_net = nn_create_config(&config);
    if (interval_net == NULL) {
        fputs("failed to allocate interval model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_net,
                                   "model interval-mlp",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_FULL,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    print_interval_metrics("model interval-mlp",
                           evaluate_interval_model(interval_net,
                                                   &sieve,
                                                   &test,
                                                   max_test_states,
                                                   EXACT_ACTION_COUNT,
                                                   (double)G_MAX,
                                                   log_scale,
                                                   &FEATURE_MASK_FULL,
                                                   candidate_inputs,
                                                   scores,
                                                   probabilities,
                                                   target_distribution));

    printf("Raw-residue interval variants remove explicit small-prime divisibility flags.\n");
    interval_raw_linear_net = nn_create_config(&linear_config);
    if (interval_raw_linear_net == NULL) {
        fputs("failed to allocate raw-residue interval linear model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_raw_linear_net,
                                   "model interval-raw-linear",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_RAW_RESIDUE,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    print_interval_metrics("model interval-raw-linear",
                           evaluate_interval_model(interval_raw_linear_net,
                                                   &sieve,
                                                   &test,
                                                   max_test_states,
                                                   EXACT_ACTION_COUNT,
                                                   (double)G_MAX,
                                                   log_scale,
                                                   &FEATURE_MASK_RAW_RESIDUE,
                                                   candidate_inputs,
                                                   scores,
                                                   probabilities,
                                                   target_distribution));

    interval_raw_net = nn_create_config(&config);
    if (interval_raw_net == NULL) {
        fputs("failed to allocate raw-residue interval model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_raw_net,
                                   "model interval-raw-mlp",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_RAW_RESIDUE,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    print_interval_metrics("model interval-raw-mlp",
                           evaluate_interval_model(interval_raw_net,
                                                   &sieve,
                                                   &test,
                                                   max_test_states,
                                                   EXACT_ACTION_COUNT,
                                                   (double)G_MAX,
                                                   log_scale,
                                                   &FEATURE_MASK_RAW_RESIDUE,
                                                   candidate_inputs,
                                                   scores,
                                                   probabilities,
                                                   target_distribution));

    nn_free(interval_raw_linear_net);
    interval_raw_linear_net = NULL;
    nn_free(interval_raw_net);
    interval_raw_net = NULL;

    printf("\n=== Wider Raw-Residue Interval Sweep ===\n");
    size_t sweep_train_states = max_train_states < 5000 ? max_train_states : 5000;
    size_t sweep_test_states = max_test_states < 5000 ? max_test_states : 5000;
    size_t sweep_epochs = supervised_epochs == 0 ? 0 : 1;
    printf("Each window trains fresh raw-residue linear and MLP scorers without explicit divisibility flags.\n");
    printf("Wide-window runs are capped at train_states=%zu test_states=%zu epochs=%zu for CLI runtime.\n",
           sweep_train_states,
           sweep_test_states,
           sweep_epochs);
    size_t interval_sweep_count = sizeof(INTERVAL_SWEEP_GAPS) / sizeof(INTERVAL_SWEEP_GAPS[0]);
    for (size_t sweep = 0; sweep < interval_sweep_count; ++sweep) {
        uint32_t window_gap = INTERVAL_SWEEP_GAPS[sweep];
        size_t candidate_count = window_gap / 2u;
        char label[96];

        printf("\n-- interval window 2..%u --\n", window_gap);
        build_interval_frequency_policy(&sieve,
                                        &fit,
                                        candidate_count,
                                        interval_frequency_ranked,
                                        interval_frequency_probabilities);

        snprintf(label, sizeof(label), "baseline interval-frequency-%u", window_gap);
        print_interval_metrics(label,
                               evaluate_interval_static_policy(&sieve,
                                                               &test,
                                                               interval_frequency_ranked,
                                                               interval_frequency_probabilities,
                                                               candidate_count,
                                                               sweep_test_states,
                                                               target_distribution));

        snprintf(label, sizeof(label), "baseline interval-wheel-frequency-%u", window_gap);
        print_interval_metrics(label,
                               evaluate_interval_wheel_frequency_policy(&sieve,
                                                                        &test,
                                                                        interval_frequency_probabilities,
                                                                        candidate_count,
                                                                        sweep_test_states,
                                                                        probabilities,
                                                                        target_distribution));

        interval_raw_linear_net = nn_create_config(&linear_config);
        if (interval_raw_linear_net == NULL) {
            fputs("failed to allocate sweep raw-residue interval linear model\n", stderr);
            ok = 0;
            goto cleanup;
        }

        snprintf(label, sizeof(label), "model interval-raw-linear-%u", window_gap);
        ok = train_interval_supervised(interval_raw_linear_net,
                                       label,
                                       &sieve,
                                       &fit,
                                       sweep_train_states,
                                       candidate_count,
                                       (double)window_gap,
                                       sweep_epochs,
                                       (nn_real)0.005,
                                       log_scale,
                                       &FEATURE_MASK_RAW_RESIDUE,
                                       candidate_inputs,
                                       target_distribution,
                                       order,
                                       &rng);
        if (!ok) {
            goto cleanup;
        }
        print_interval_metrics(label,
                               evaluate_interval_model(interval_raw_linear_net,
                                                       &sieve,
                                                       &test,
                                                       sweep_test_states,
                                                       candidate_count,
                                                       (double)window_gap,
                                                       log_scale,
                                                       &FEATURE_MASK_RAW_RESIDUE,
                                                       candidate_inputs,
                                                       scores,
                                                       probabilities,
                                                       target_distribution));
        nn_free(interval_raw_linear_net);
        interval_raw_linear_net = NULL;

        interval_raw_net = nn_create_config(&config);
        if (interval_raw_net == NULL) {
            fputs("failed to allocate sweep raw-residue interval model\n", stderr);
            ok = 0;
            goto cleanup;
        }

        snprintf(label, sizeof(label), "model interval-raw-mlp-%u", window_gap);
        ok = train_interval_supervised(interval_raw_net,
                                       label,
                                       &sieve,
                                       &fit,
                                       sweep_train_states,
                                       candidate_count,
                                       (double)window_gap,
                                       sweep_epochs,
                                       (nn_real)0.005,
                                       log_scale,
                                       &FEATURE_MASK_RAW_RESIDUE,
                                       candidate_inputs,
                                       target_distribution,
                                       order,
                                       &rng);
        if (!ok) {
            goto cleanup;
        }
        print_interval_metrics(label,
                               evaluate_interval_model(interval_raw_net,
                                                       &sieve,
                                                       &test,
                                                       sweep_test_states,
                                                       candidate_count,
                                                       (double)window_gap,
                                                       log_scale,
                                                       &FEATURE_MASK_RAW_RESIDUE,
                                                       candidate_inputs,
                                                       scores,
                                                       probabilities,
                                                       target_distribution));
        nn_free(interval_raw_net);
        interval_raw_net = NULL;
    }

    printf("\n=== Entropy-Regularized Training ===\n");
    entropy_net = nn_create_config(&config);
    if (entropy_net == NULL) {
        fputs("failed to allocate entropy model\n", stderr);
        ok = 0;
        goto cleanup;
    }

    ok = train_supervised(entropy_net,
                          "model entropy-supervised",
                          &fit,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          SUPERVISED_ENTROPY_WEIGHT,
                          0,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    entropy_metrics = evaluate_model(entropy_net,
                                     &sieve,
                                     &test,
                                     max_test_states,
                                     log_scale,
                                     &FEATURE_MASK_FULL,
                                     1.0,
                                     0,
                                     0,
                                     candidate_inputs,
                                     scores,
                                     probabilities);
    print_metrics("model entropy-supervised", entropy_metrics);

    entropy_temperature = tune_temperature(entropy_net,
                                           &sieve,
                                           &calibration,
                                           max_train_states,
                                           log_scale,
                                           &FEATURE_MASK_FULL,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
    entropy_temperature_metrics = evaluate_model(entropy_net,
                                                 &sieve,
                                                 &test,
                                                 max_test_states,
                                                 log_scale,
                                                 &FEATURE_MASK_FULL,
                                                 entropy_temperature,
                                                 0,
                                                 0,
                                                 candidate_inputs,
                                                 scores,
                                                 probabilities);
    print_metrics("model entropy-supervised temp-scaled", entropy_temperature_metrics);

    printf("\n=== Strict Verifier RL ===\n");
    if (rl_epochs == 0) {
        printf("skipped because rl_epochs=0\n");
    } else {
        ok = train_strict_verifier_rl(net,
                                      "model strict-rl",
                                      &fit,
                                      max_train_states,
                                      rl_epochs,
                                      (nn_real)0.001,
                                      log_scale,
                                      &FEATURE_MASK_FULL,
                                      candidate_inputs,
                                      scores,
                                      probabilities,
                                      order,
                                      &rng);
        if (!ok) {
            goto cleanup;
        }

        strict_rl_metrics = evaluate_model(net,
                                           &sieve,
                                           &test,
                                           max_test_states,
                                           log_scale,
                                           &FEATURE_MASK_FULL,
                                           1.0,
                                           0,
                                           1,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
        have_strict_rl_metrics = 1;
        print_metrics("model strict-rl", strict_rl_metrics);

        strict_rl_temperature = tune_temperature(net,
                                                 &sieve,
                                                 &calibration,
                                                 max_train_states,
                                                 log_scale,
                                                 &FEATURE_MASK_FULL,
                                                 candidate_inputs,
                                                 scores,
                                                 probabilities);
        strict_rl_temperature_metrics = evaluate_model(net,
                                                       &sieve,
                                                       &test,
                                                       max_test_states,
                                                       log_scale,
                                                       &FEATURE_MASK_FULL,
                                                       strict_rl_temperature,
                                                       0,
                                                       0,
                                                       candidate_inputs,
                                                       scores,
                                                       probabilities);
        print_metrics("model strict-rl temp-scaled", strict_rl_temperature_metrics);
    }

    if (have_strict_rl_metrics) {
        print_calibration_comparison("calibration comparison raw",
                                     supervised_metrics,
                                     entropy_metrics,
                                     strict_rl_metrics);
        print_calibration_comparison("calibration comparison temp-scaled",
                                     supervised_temperature_metrics,
                                     entropy_temperature_metrics,
                                     strict_rl_temperature_metrics);
    }

cleanup:
    free(order);
    free(target_distribution);
    free(probabilities);
    free(scores);
    free(candidate_inputs);
    nn_free(interval_raw_linear_net);
    nn_free(interval_raw_net);
    nn_free(interval_linear_net);
    nn_free(interval_net);
    nn_free(wheel_net);
    nn_free(entropy_net);
    nn_free(net);
    free_conditioned_policy(&residue_policy);
    free_gap_dataset(&train);
    free_gap_dataset(&fit);
    free_gap_dataset(&calibration);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
