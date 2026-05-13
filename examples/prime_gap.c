#include "nn.h"

#include <errno.h>
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
};

static const double DEFAULT_REGULARIZED_RL_REFERENCE_WEIGHT = 1.20;
static const double DEFAULT_REGULARIZED_RL_ENTROPY_WEIGHT = 0.12;
static const uint32_t FEATURE_MODS[FEATURE_MOD_COUNT] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};

typedef struct {
    const char *name;
    int use_log;
    int use_gap;
    int use_overflow;
    int use_residue;
} FeatureMask;

static const FeatureMask FEATURE_MASK_FULL = {"full", 1, 1, 1, 1};
static const FeatureMask FEATURE_MASK_LOG_ONLY = {"log-only", 1, 0, 0, 0};
static const FeatureMask FEATURE_MASK_GAP_ONLY = {"gap-only", 0, 1, 1, 0};
static const FeatureMask FEATURE_MASK_RESIDUE_ONLY = {"residue-only", 0, 0, 1, 1};

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
    size_t train_limit_arg;
    size_t test_limit_arg;
    size_t supervised_epochs;
    size_t rl_epochs;
    size_t max_train_states;
    size_t max_test_states;
    double regularized_reference_weight;
    double regularized_entropy_weight;
} PrimeGapRunConfig;

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

static int parse_size_arg(int argc, char **argv, int index, size_t fallback, size_t *value_out)
{
    if (index >= argc) {
        *value_out = fallback;
        return 1;
    }

    const char *text = argv[index];
    if (text[0] == '\0' || text[0] == '-' || text[0] == '+') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE ||
        value > (unsigned long long)SIZE_MAX) {
        return 0;
    }

    *value_out = (size_t)value;
    return 1;
}

static int parse_double_arg(int argc, char **argv, int index, double fallback, double *value_out)
{
    if (index >= argc) {
        *value_out = fallback;
        return 1;
    }

    const char *text = argv[index];
    if (text[0] == '\0') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !isfinite(value)) {
        return 0;
    }

    *value_out = value;
    return 1;
}

static PrimeGapRunConfig default_run_config(void)
{
    PrimeGapRunConfig config;
    config.train_limit_arg = DEFAULT_TRAIN_LIMIT;
    config.test_limit_arg = DEFAULT_TEST_LIMIT;
    config.supervised_epochs = DEFAULT_SUPERVISED_EPOCHS;
    config.rl_epochs = DEFAULT_RL_EPOCHS;
    config.max_train_states = DEFAULT_MAX_TRAIN_STATES;
    config.max_test_states = DEFAULT_MAX_TEST_STATES;
    config.regularized_reference_weight = DEFAULT_REGULARIZED_RL_REFERENCE_WEIGHT;
    config.regularized_entropy_weight = DEFAULT_REGULARIZED_RL_ENTROPY_WEIGHT;
    return config;
}

static int parse_run_config(int argc, char **argv, PrimeGapRunConfig *config)
{
    *config = default_run_config();
    return parse_size_arg(argc, argv, 1, DEFAULT_TRAIN_LIMIT, &config->train_limit_arg) &&
           parse_size_arg(argc, argv, 2, DEFAULT_TEST_LIMIT, &config->test_limit_arg) &&
           parse_size_arg(argc, argv, 3, DEFAULT_SUPERVISED_EPOCHS, &config->supervised_epochs) &&
           parse_size_arg(argc, argv, 4, DEFAULT_RL_EPOCHS, &config->rl_epochs) &&
           parse_size_arg(argc, argv, 5, DEFAULT_MAX_TRAIN_STATES, &config->max_train_states) &&
           parse_size_arg(argc, argv, 6, DEFAULT_MAX_TEST_STATES, &config->max_test_states) &&
           parse_double_arg(argc,
                            argv,
                            7,
                            DEFAULT_REGULARIZED_RL_REFERENCE_WEIGHT,
                            &config->regularized_reference_weight) &&
           parse_double_arg(argc,
                            argv,
                            8,
                            DEFAULT_REGULARIZED_RL_ENTROPY_WEIGHT,
                            &config->regularized_entropy_weight);
}

static int run_config_is_valid(const PrimeGapRunConfig *config)
{
    return config->test_limit_arg > config->train_limit_arg &&
           config->train_limit_arg >= 16 &&
           config->test_limit_arg <= (size_t)UINT32_MAX - (size_t)G_MAX &&
           config->train_limit_arg <= (size_t)UINT32_MAX &&
           config->regularized_reference_weight >= 0.0 &&
           config->regularized_entropy_weight >= 0.0;
}

static void print_usage(FILE *stream)
{
    fputs("usage: prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] "
          "[max_train_states] [max_test_states] [regularized_reference_weight] "
          "[regularized_entropy_weight]\n",
          stream);
}

static uint32_t validation_start_limit(uint32_t train_limit)
{
    uint32_t holdout = train_limit / 5u;
    if (holdout == 0 || holdout >= train_limit - 3u) {
        return train_limit;
    }
    return train_limit - holdout;
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
                              GapDataset *validation,
                              GapDataset *test)
{
    memset(train, 0, sizeof(*train));
    memset(validation, 0, sizeof(*validation));
    memset(test, 0, sizeof(*test));

    uint32_t validation_start = validation_start_limit(train_limit);

    for (size_t i = 0; i + 1 < sieve->prime_count; ++i) {
        uint32_t p = sieve->primes[i];
        uint32_t next = sieve->primes[i + 1];
        uint32_t gap = next - p;

        if (p < 3 || (gap & 1u) != 0) {
            continue;
        }

        GapDataset *target = NULL;
        if (p < validation_start) {
            target = train;
        } else if (p < train_limit) {
            target = validation;
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
            free_gap_dataset(validation);
            free_gap_dataset(test);
            return 0;
        }
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

static void fill_order(size_t *order, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        order[i] = i;
    }
}

static size_t action_gap(size_t action)
{
    if (action == OVERFLOW_ACTION) {
        return G_MAX + 2;
    }

    return (action + 1) * 2;
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

static void fill_candidate_features_masked(nn_real *inputs,
                                           uint32_t p,
                                           double log_scale,
                                           const FeatureMask *mask)
{
    if (mask == NULL) {
        mask = &FEATURE_MASK_FULL;
    }

    double log_p = log((double)p) / log_scale;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        nn_real *row = &inputs[action * FEATURE_COUNT];

        row[0] = mask->use_log ? (nn_real)log_p : (nn_real)0;
        row[1] = mask->use_gap ? (nn_real)((double)action_gap(action) / (double)G_MAX)
                               : (nn_real)0;
        row[2] = mask->use_overflow && action_is_overflow(action) ? (nn_real)1 : (nn_real)0;
        if (action_is_overflow(action)) {
            for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
                row[3 + i] = 0;
            }
            continue;
        }

        uint32_t gap = (uint32_t)action_gap(action);
        uint32_t candidate = p + gap;
        for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
            uint32_t mod = FEATURE_MODS[i];
            row[3 + i] = mask->use_residue && candidate > mod && candidate % mod == 0
                             ? (nn_real)1
                             : (nn_real)0;
        }
    }
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

static uint32_t extend_segmented_first_prime_gap(const Sieve *sieve,
                                                 uint32_t p,
                                                 uint32_t from_gap,
                                                 uint32_t to_gap,
                                                 uint8_t *composite,
                                                 BlockWork *work)
{
    uint32_t old_candidate_count = from_gap / 2u;
    uint32_t candidate_count = to_gap / 2u;
    uint64_t start_number = (uint64_t)p + (uint64_t)from_gap + 2u;
    uint64_t end_number = (uint64_t)p + (uint64_t)to_gap;

    memset(work, 0, sizeof(*work));
    if (to_gap <= from_gap) {
        return 0;
    }

    work->candidates = candidate_count - old_candidate_count;

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
            uint32_t gap = (uint32_t)(value - (uint64_t)p);
            if (gap < 2 || (gap & 1u) != 0) {
                continue;
            }

            uint32_t index = gap / 2u - 1u;
            if (index >= old_candidate_count && index < candidate_count && !composite[index]) {
                composite[index] = 1;
                ++work->marks;
            }
        }
    }

    for (uint32_t index = old_candidate_count; index < candidate_count; ++index) {
        ++work->survivor_checks;
        if (!composite[index]) {
            return (index + 1u) * 2u;
        }
    }

    return 0;
}

static BlockWork assisted_block_work(const Sieve *sieve,
                                     uint32_t p,
                                     uint32_t true_gap,
                                     const size_t *ranked,
                                     size_t top_k)
{
    BlockWork total = {0, 0, 0, 0};
    uint32_t max_gap = true_gap;
    uint32_t verified_gap = 0;
    uint32_t candidate_count;
    uint8_t local_composite[EXACT_ACTION_COUNT];
    uint8_t *composite = local_composite;

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        if (!action_is_overflow(proposed) && action_gap(proposed) > max_gap) {
            max_gap = (uint32_t)action_gap(proposed);
        }
    }

    candidate_count = max_gap / 2u;
    if (candidate_count == 0) {
        return total;
    }

    if (candidate_count <= EXACT_ACTION_COUNT) {
        memset(local_composite, 0, candidate_count * sizeof(*local_composite));
    } else {
        composite = calloc(candidate_count, sizeof(*composite));
        if (composite == NULL) {
            return block_scan_work(sieve, p, true_gap);
        }
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        BlockWork part;
        uint32_t found_gap;

        if (action_is_overflow(proposed)) {
            break;
        }

        uint32_t proposed_gap = (uint32_t)action_gap(proposed);
        if (proposed_gap <= verified_gap) {
            continue;
        }

        found_gap =
            extend_segmented_first_prime_gap(sieve, p, verified_gap, proposed_gap, composite, &part);
        add_block_work(&total, part);
        verified_gap = proposed_gap;

        if (found_gap != 0) {
            if (composite != local_composite) {
                free(composite);
            }
            return total;
        }
    }

    if (true_gap > verified_gap) {
        BlockWork part;
        extend_segmented_first_prime_gap(sieve, p, verified_gap, true_gap, composite, &part);
        add_block_work(&total, part);
    }

    if (composite != local_composite) {
        free(composite);
    }
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

static void softmax_temperature(const nn_real *scores,
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

static double distribution_entropy(const nn_real *probabilities, size_t count)
{
    double entropy = 0.0;
    for (size_t i = 0; i < count; ++i) {
        double probability = safe_probability((double)probabilities[i]);
        entropy -= probability * log(probability);
    }
    return entropy;
}

static double distribution_kl(const nn_real *reference,
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

        fill_candidate_features_masked(candidate_inputs, p, log_scale, mask);
        nn_forward_batch(net, candidate_inputs, scores, ACTION_COUNT);
        softmax_temperature(scores, probabilities, ACTION_COUNT, temperature);
        rank_actions(scores, ranked);
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
        nn_forward_batch(net, candidate_inputs, scores, ACTION_COUNT);
        softmax_temperature(scores, probabilities, ACTION_COUNT, temperature);
        nll += -log(safe_probability((double)probabilities[target]));
    }

    return nll / (double)samples;
}

static double fit_temperature(NeuralNet *net,
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
    finalize_eval_metrics(&metrics);
    return metrics;
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

static void print_comparison_header(void)
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

static void print_comparison_row(const char *label, double temperature, EvalMetrics metrics)
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

static int train_supervised(NeuralNet *net,
                            const char *label,
                            const GapDataset *dataset,
                            size_t max_states,
                            size_t epochs,
                            nn_real learning_rate,
                            double log_scale,
                            const FeatureMask *mask,
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
            fill_candidate_features_masked(candidate_inputs, dataset->prime[index], log_scale, mask);
            nn_real step_loss = nn_train_softmax_scorer(net,
                                                        candidate_inputs,
                                                        ACTION_COUNT,
                                                        gap_to_action(dataset->gap[index]),
                                                        learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu loss=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               loss / (double)train_steps,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
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

static int train_regularized_verifier_rl(NeuralNet *net,
                                         NeuralNet *reference_net,
                                         const char *label,
                                         const GapDataset *dataset,
                                         size_t max_states,
                                         size_t epochs,
                                         nn_real learning_rate,
                                         nn_real reference_weight,
                                         nn_real entropy_weight,
                                         double log_scale,
                                         const FeatureMask *mask,
                                         nn_real *candidate_inputs,
                                         nn_real *scores,
                                         nn_real *probabilities,
                                         nn_real *reference_scores,
                                         nn_real *reference_probabilities,
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
        double mean_entropy = 0.0;
        double mean_reference_kl = 0.0;
        size_t exact_hits = 0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            size_t target = gap_to_action(dataset->gap[index]);

            fill_candidate_features_masked(candidate_inputs, dataset->prime[index], log_scale, mask);
            nn_forward_batch(net, candidate_inputs, scores, ACTION_COUNT);
            nn_softmax(scores, probabilities, ACTION_COUNT);
            nn_forward_batch(reference_net, candidate_inputs, reference_scores, ACTION_COUNT);
            nn_softmax(reference_scores, reference_probabilities, ACTION_COUNT);

            size_t selected = sample_action(probabilities, rng);
            double reward = selected == target ? 1.0 : 0.0;
            double advantage = reward - baseline;

            baseline += 0.01 * (reward - baseline);
            exact_hits += reward > 0.0 ? 1u : 0u;
            mean_entropy += distribution_entropy(probabilities, ACTION_COUNT);
            mean_reference_kl +=
                distribution_kl(reference_probabilities, probabilities, ACTION_COUNT);

            nn_real step_loss = nn_train_softmax_scorer_policy_regularized_cached(
                net,
                ACTION_COUNT,
                selected,
                (nn_real)advantage,
                reference_probabilities,
                reference_weight,
                entropy_weight,
                learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            objective += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu sampled_exact=%.2f%% baseline=%.3f objective=%.4f "
               "entropy=%.3f ref_kl=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               100.0 * (double)exact_hits / (double)train_steps,
               baseline,
               objective / (double)train_steps,
               mean_entropy / (double)train_steps,
               mean_reference_kl / (double)train_steps,
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
                                     0,
                                     1.0,
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
    PrimeGapRunConfig run_config;
    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    uint32_t rng = 20240511u;
    int ok = 1;

    if (!parse_run_config(argc, argv, &run_config) || !run_config_is_valid(&run_config)) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }

    uint32_t train_limit = (uint32_t)run_config.train_limit_arg;
    uint32_t test_limit = (uint32_t)run_config.test_limit_arg;
    uint32_t sieve_limit = test_limit + (uint32_t)G_MAX;
    uint32_t validation_start = validation_start_limit(train_limit);
    size_t max_validation_states = run_config.max_test_states;

    printf("prime-gap scorer train_limit=%u validation_start=%u test_limit=%u "
           "g_max=%d actions=%d features=%d\n",
           train_limit,
           validation_start,
           test_limit,
           G_MAX,
           ACTION_COUNT,
           FEATURE_COUNT);
    printf("regularized_rl reference_weight=%.3f entropy_weight=%.3f\n",
           run_config.regularized_reference_weight,
           run_config.regularized_entropy_weight);

    double started = now_seconds();
    if (!build_sieve(sieve_limit, &sieve)) {
        fputs("failed to build sieve\n", stderr);
        return EXIT_FAILURE;
    }
    printf("sieve limit=%u primes=%zu time=%.3fs\n",
           sieve_limit,
           sieve.prime_count,
           now_seconds() - started);

    if (!build_gap_datasets(&sieve, train_limit, test_limit, &train, &validation, &test)) {
        fputs("failed to build gap datasets\n", stderr);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    printf("dataset train=%zu exact_action_coverage=%.2f%% "
           "validation=%zu exact_action_coverage=%.2f%% "
           "test=%zu exact_action_coverage=%.2f%% "
           "overflow_actions train=%zu validation=%zu test=%zu\n",
           train.count,
           train.total_seen == 0
               ? 0.0
               : 100.0 * (double)(train.total_seen - train.overflow_count) /
                     (double)train.total_seen,
           validation.count,
           validation.total_seen == 0
               ? 0.0
               : 100.0 * (double)(validation.total_seen - validation.overflow_count) /
                     (double)validation.total_seen,
           test.count,
           test.total_seen == 0
               ? 0.0
               : 100.0 * (double)(test.total_seen - test.overflow_count) /
                     (double)test.total_seen,
           train.overflow_count,
           validation.overflow_count,
           test.overflow_count);

    if (train.count == 0 || validation.count == 0 || test.count == 0) {
        fputs("empty train, validation, or test split\n", stderr);
        free_gap_dataset(&train);
        free_gap_dataset(&validation);
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

    NeuralNet *net = nn_create_config(&config);
    NeuralNet *strict_net = NULL;
    NeuralNet *regularized_net = NULL;
    nn_real *candidate_inputs = calloc((size_t)ACTION_COUNT * FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(ACTION_COUNT, sizeof(*probabilities));
    nn_real *reference_scores = calloc(ACTION_COUNT, sizeof(*reference_scores));
    nn_real *reference_probabilities = calloc(ACTION_COUNT, sizeof(*reference_probabilities));
    size_t *order = calloc(train.count, sizeof(*order));
    if (net == NULL || candidate_inputs == NULL || scores == NULL || probabilities == NULL ||
        reference_scores == NULL || reference_probabilities == NULL || order == NULL) {
        fputs("failed to allocate model buffers\n", stderr);
        ok = 0;
        goto cleanup;
    }

    fill_order(order, train.count);

    double log_scale = log((double)test_limit + (double)G_MAX);
    size_t frequency_ranked[ACTION_COUNT];
    nn_real frequency_probabilities[ACTION_COUNT];

    printf("\n=== Baselines ===\n");
    build_frequency_policy(&train, frequency_ranked, frequency_probabilities);
    print_metrics("baseline empirical-gap",
                  evaluate_static_policy(&sieve,
                                         &test,
                                         frequency_ranked,
                                         frequency_probabilities,
                                         run_config.max_test_states,
                                         1));

    print_metrics("model before",
                  evaluate_model(net,
                                 &sieve,
                                 &test,
                                 run_config.max_test_states,
                                 log_scale,
                                 &FEATURE_MASK_FULL,
                                 1,
                                 1.0,
                                 candidate_inputs,
                                 scores,
                                 probabilities));

    printf("\n=== Feature Ablations ===\n");
    printf("Supervised-only runs; verifier diagnostics skipped to keep these capability-focused.\n");
    ok = run_supervised_ablation("ablation log-only",
                                 &FEATURE_MASK_LOG_ONLY,
                                 &config,
                                 &sieve,
                                 &train,
                                 &test,
                                 run_config.max_train_states,
                                 run_config.max_test_states,
                                 run_config.supervised_epochs,
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
                                 &train,
                                 &test,
                                 run_config.max_train_states,
                                 run_config.max_test_states,
                                 run_config.supervised_epochs,
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
                                 &train,
                                 &test,
                                 run_config.max_train_states,
                                 run_config.max_test_states,
                                 run_config.supervised_epochs,
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
                          &train,
                          run_config.max_train_states,
                          run_config.supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    double supervised_temperature_nll = 0.0;
    double supervised_temperature = fit_temperature(net,
                                                    &validation,
                                                    max_validation_states,
                                                    log_scale,
                                                    &FEATURE_MASK_FULL,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities,
                                                    &supervised_temperature_nll);
    printf("model supervised temperature=%.2f validation_nll=%.4f\n",
           supervised_temperature,
           supervised_temperature_nll);

    EvalMetrics supervised_metrics = evaluate_model(net,
                                                    &sieve,
                                                    &test,
                                                    run_config.max_test_states,
                                                    log_scale,
                                                    &FEATURE_MASK_FULL,
                                                    1,
                                                    1.0,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
    EvalMetrics supervised_temp_metrics = evaluate_model(net,
                                                         &sieve,
                                                         &test,
                                                         run_config.max_test_states,
                                                         log_scale,
                                                         &FEATURE_MASK_FULL,
                                                         0,
                                                         supervised_temperature,
                                                         candidate_inputs,
                                                         scores,
                                                         probabilities);
    print_metrics("model supervised", supervised_metrics);
    print_metrics("model supervised temp-scaled", supervised_temp_metrics);

    printf("\n=== Strict Verifier RL ===\n");
    int have_rl_metrics = 0;
    double rl_temperature = 1.0;
    EvalMetrics strict_metrics;
    EvalMetrics strict_temp_metrics;
    double regularized_temperature = 1.0;
    EvalMetrics regularized_metrics;
    EvalMetrics regularized_temp_metrics;

    if (run_config.rl_epochs == 0) {
        printf("skipped because rl_epochs=0\n");
    } else {
        strict_net = nn_clone(net);
        regularized_net = nn_clone(net);
        if (strict_net == NULL || regularized_net == NULL) {
            fputs("failed to clone supervised model\n", stderr);
            ok = 0;
            goto cleanup;
        }

        uint32_t strict_rng = rng;
        uint32_t regularized_rng = rng;

        fill_order(order, train.count);
        ok = train_strict_verifier_rl(strict_net,
                                      "model strict-rl",
                                      &train,
                                      run_config.max_train_states,
                                      run_config.rl_epochs,
                                      (nn_real)0.001,
                                      log_scale,
                                      &FEATURE_MASK_FULL,
                                      candidate_inputs,
                                      scores,
                                      probabilities,
                                      order,
                                      &strict_rng);
        if (!ok) {
            goto cleanup;
        }

        double rl_temperature_nll = 0.0;
        rl_temperature = fit_temperature(strict_net,
                                         &validation,
                                         max_validation_states,
                                         log_scale,
                                         &FEATURE_MASK_FULL,
                                         candidate_inputs,
                                         scores,
                                         probabilities,
                                         &rl_temperature_nll);
        printf("model strict-rl temperature=%.2f validation_nll=%.4f\n",
               rl_temperature,
               rl_temperature_nll);

        strict_metrics = evaluate_model(strict_net,
                                        &sieve,
                                        &test,
                                        run_config.max_test_states,
                                        log_scale,
                                        &FEATURE_MASK_FULL,
                                        1,
                                        1.0,
                                        candidate_inputs,
                                        scores,
                                        probabilities);
        strict_temp_metrics = evaluate_model(strict_net,
                                             &sieve,
                                             &test,
                                             run_config.max_test_states,
                                             log_scale,
                                             &FEATURE_MASK_FULL,
                                             0,
                                             rl_temperature,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        print_metrics("model strict-rl", strict_metrics);
        print_metrics("model strict-rl temp-scaled", strict_temp_metrics);

        printf("\n=== Regularized Verifier RL ===\n");
        printf("reference_weight=%.3f entropy_weight=%.3f\n",
               run_config.regularized_reference_weight,
               run_config.regularized_entropy_weight);
        fill_order(order, train.count);
        ok = train_regularized_verifier_rl(regularized_net,
                                           net,
                                           "model regularized-rl",
                                           &train,
                                           run_config.max_train_states,
                                           run_config.rl_epochs,
                                           (nn_real)0.001,
                                           (nn_real)run_config.regularized_reference_weight,
                                           (nn_real)run_config.regularized_entropy_weight,
                                           log_scale,
                                           &FEATURE_MASK_FULL,
                                           candidate_inputs,
                                           scores,
                                           probabilities,
                                           reference_scores,
                                           reference_probabilities,
                                           order,
                                           &regularized_rng);
        if (!ok) {
            goto cleanup;
        }

        double regularized_temperature_nll = 0.0;
        regularized_temperature = fit_temperature(regularized_net,
                                                  &validation,
                                                  max_validation_states,
                                                  log_scale,
                                                  &FEATURE_MASK_FULL,
                                                  candidate_inputs,
                                                  scores,
                                                  probabilities,
                                                  &regularized_temperature_nll);
        printf("model regularized-rl temperature=%.2f validation_nll=%.4f\n",
               regularized_temperature,
               regularized_temperature_nll);

        regularized_metrics = evaluate_model(regularized_net,
                                             &sieve,
                                             &test,
                                             run_config.max_test_states,
                                             log_scale,
                                             &FEATURE_MASK_FULL,
                                             1,
                                             1.0,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        regularized_temp_metrics = evaluate_model(regularized_net,
                                                  &sieve,
                                                  &test,
                                                  run_config.max_test_states,
                                                  log_scale,
                                                  &FEATURE_MASK_FULL,
                                                  0,
                                                  regularized_temperature,
                                                  candidate_inputs,
                                                  scores,
                                                  probabilities);
        print_metrics("model regularized-rl", regularized_metrics);
        print_metrics("model regularized-rl temp-scaled", regularized_temp_metrics);
        have_rl_metrics = 1;
    }

    print_comparison_header();
    print_comparison_row("supervised", 1.0, supervised_metrics);
    print_comparison_row("supervised temp-scaled",
                         supervised_temperature,
                         supervised_temp_metrics);
    if (have_rl_metrics) {
        print_comparison_row("strict-rl", 1.0, strict_metrics);
        print_comparison_row("strict-rl temp-scaled", rl_temperature, strict_temp_metrics);
        print_comparison_row("regularized-rl", 1.0, regularized_metrics);
        print_comparison_row("regularized-rl temp-scaled",
                             regularized_temperature,
                             regularized_temp_metrics);
    }

cleanup:
    free(order);
    free(reference_probabilities);
    free(reference_scores);
    free(probabilities);
    free(scores);
    free(candidate_inputs);
    nn_free(regularized_net);
    nn_free(strict_net);
    nn_free(net);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif
