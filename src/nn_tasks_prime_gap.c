#include "nn.h"
#include "nn_tasks_prime_gap.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef NN_PRIME_GAP_G_MAX
#define NN_PRIME_GAP_G_MAX 256
#endif

#ifndef NN_PRIME_GAP_FEATURE_MOD_COUNT
#define NN_PRIME_GAP_FEATURE_MOD_COUNT 10
#endif

#ifndef NN_PRIME_GAP_HIDDEN1
#define NN_PRIME_GAP_HIDDEN1 32
#endif

#ifndef NN_PRIME_GAP_HIDDEN2
#define NN_PRIME_GAP_HIDDEN2 16
#endif

#ifndef NN_SYMBOLIC_RULE_MAX_MOD
#define NN_SYMBOLIC_RULE_MAX_MOD 255
#endif

#ifndef NN_SYMBOLIC_MAX_LEARNED_RULES
#define NN_SYMBOLIC_MAX_LEARNED_RULES 64
#endif

enum {
    G_MAX = NN_PRIME_GAP_G_MAX,
    EXACT_ACTION_COUNT = G_MAX / 2,
    OVERFLOW_ACTION = EXACT_ACTION_COUNT,
    ACTION_COUNT = EXACT_ACTION_COUNT + 1,
    FEATURE_MOD_COUNT = NN_PRIME_GAP_FEATURE_MOD_COUNT,
    FEATURE_COUNT = 3 + FEATURE_MOD_COUNT,
    CONDITION_FEATURE_MOD_COUNT = FEATURE_MOD_COUNT <= 16 ? FEATURE_MOD_COUNT : 16,
    CONDITION_SIGNATURE_COUNT = (1 << CONDITION_FEATURE_MOD_COUNT) + 1,
    DEFAULT_TRAIN_LIMIT = 1000000,
    DEFAULT_TEST_LIMIT = 3000000,
    DEFAULT_SUPERVISED_EPOCHS = 2,
    DEFAULT_RL_EPOCHS = 1,
    DEFAULT_MAX_TRAIN_STATES = 30000,
    DEFAULT_MAX_TEST_STATES = 30000,
    CALIBRATION_BINS = 10,
    PRIME_BUCKET_COUNT = 4,
    SYMBOLIC_RULE_MOD_MIN = 3,
    SYMBOLIC_RULE_FEATURE_COUNT = 5,
    SYMBOLIC_RULE_SCORER_EPOCHS = 20,
};

static const double DEFAULT_REGULARIZED_RL_REFERENCE_WEIGHT = 1.20;
static const double DEFAULT_REGULARIZED_RL_ENTROPY_WEIGHT = 0.12;
static const uint32_t FEATURE_MODS[] = {
    3,  5,  7,  11, 13, 17, 19, 23, 29, 31, 37, 41,
    43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157,
    163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227,
    229, 233, 239, 241, 251,
};

_Static_assert(NN_PRIME_GAP_FEATURE_MOD_COUNT > 0,
               "NN_PRIME_GAP_FEATURE_MOD_COUNT must be positive");
_Static_assert(NN_PRIME_GAP_FEATURE_MOD_COUNT <=
                   sizeof(FEATURE_MODS) / sizeof(FEATURE_MODS[0]),
               "not enough configured prime-gap residue moduli");
_Static_assert(NN_PRIME_GAP_G_MAX >= 2 && NN_PRIME_GAP_G_MAX % 2 == 0,
               "NN_PRIME_GAP_G_MAX must be an even value of at least 2");
_Static_assert(NN_PRIME_GAP_HIDDEN1 > 0 && NN_PRIME_GAP_HIDDEN2 > 0,
               "prime-gap hidden layer sizes must be positive");
_Static_assert(NN_SYMBOLIC_RULE_MAX_MOD >= SYMBOLIC_RULE_MOD_MIN,
               "NN_SYMBOLIC_RULE_MAX_MOD must be at least 3");
_Static_assert(NN_SYMBOLIC_MAX_LEARNED_RULES > 0,
               "NN_SYMBOLIC_MAX_LEARNED_RULES must be positive");

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
    size_t *action_counts;
    size_t condition_totals[CONDITION_SIGNATURE_COUNT];
} ConditionedFrequencyPolicy;

typedef struct {
    size_t candidates;
    size_t mod_tests;
    size_t exact_tests;
} VerifierWork;

typedef struct {
    uint32_t modulus;
    uint32_t residue;
    size_t train_rejects;
    size_t validation_rejects;
    nn_real model_score;
    int domain_valid;
} SymbolicRejectRule;

typedef struct {
    SymbolicRejectRule *items;
    size_t count;
    size_t capacity;
} SymbolicRuleSet;

typedef struct {
    size_t proposed_rules;
    size_t domain_valid_rules;
    size_t accepted_rules;
    size_t train_rejects;
    size_t validation_rejects;
    double proposer_loss;
    size_t samples;
    size_t test_false_rejections;
    double direct_exact_tests;
    double hand_wheel_mod_tests;
    double hand_wheel_exact_tests;
    double learned_mod_tests;
    double learned_exact_tests;
} SymbolicWheelMetrics;

static size_t eval_sample_index(const GapDataset *dataset, size_t sample, size_t samples);
static int compare_symbolic_rules_canonical(const void *left, const void *right);

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
    double scorer_seconds;
    double verifier_seconds;
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
    size_t rng_seed_arg;
    const char *trace_path;
    const char *save_path;
    const char *load_model_path;
    const char *save_rules_path;
    const char *load_rules_path;
    const char *save_learned_model_path;
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
    config.rng_seed_arg = 20240511u;
    config.trace_path = NULL;
    config.save_path = NULL;
    config.load_model_path = NULL;
    config.save_rules_path = NULL;
    config.load_rules_path = NULL;
    config.save_learned_model_path = NULL;
    return config;
}

static int parse_run_config(int argc, char **argv, PrimeGapRunConfig *config)
{
    char *positionals[10];
    int positional_count = 1;

    *config = default_run_config();
    positionals[0] = argv[0];

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                return 0;
            }
            config->trace_path = argv[i];
        } else if (strcmp(argv[i], "--save") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                return 0;
            }
            config->save_path = argv[i];
        } else if (strcmp(argv[i], "--model") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                return 0;
            }
            config->load_model_path = argv[i];
        } else if (strcmp(argv[i], "--save-rules") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                return 0;
            }
            config->save_rules_path = argv[i];
        } else if (strcmp(argv[i], "--rules") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                return 0;
            }
            config->load_rules_path = argv[i];
        } else if (strcmp(argv[i], "--save-learned-model") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                return 0;
            }
            config->save_learned_model_path = argv[i];
        } else if (strcmp(argv[i], "--help") == 0) {
            return 0;
        } else if (positional_count < (int)(sizeof(positionals) / sizeof(positionals[0]))) {
            positionals[positional_count++] = argv[i];
        } else {
            return 0;
        }
    }

    return parse_size_arg(positional_count,
                          positionals,
                          1,
                          DEFAULT_TRAIN_LIMIT,
                          &config->train_limit_arg) &&
           parse_size_arg(positional_count,
                          positionals,
                          2,
                          DEFAULT_TEST_LIMIT,
                          &config->test_limit_arg) &&
           parse_size_arg(positional_count,
                          positionals,
                          3,
                          DEFAULT_SUPERVISED_EPOCHS,
                          &config->supervised_epochs) &&
           parse_size_arg(positional_count,
                          positionals,
                          4,
                          DEFAULT_RL_EPOCHS,
                          &config->rl_epochs) &&
           parse_size_arg(positional_count,
                          positionals,
                          5,
                          DEFAULT_MAX_TRAIN_STATES,
                          &config->max_train_states) &&
           parse_size_arg(positional_count,
                          positionals,
                          6,
                          DEFAULT_MAX_TEST_STATES,
                          &config->max_test_states) &&
           parse_double_arg(positional_count,
                            positionals,
                            7,
                            DEFAULT_REGULARIZED_RL_REFERENCE_WEIGHT,
                            &config->regularized_reference_weight) &&
           parse_double_arg(positional_count,
                            positionals,
                            8,
                            DEFAULT_REGULARIZED_RL_ENTROPY_WEIGHT,
                            &config->regularized_entropy_weight) &&
           parse_size_arg(positional_count, positionals, 9, 20240511u, &config->rng_seed_arg);
}

static int run_config_is_valid(const PrimeGapRunConfig *config)
{
    return config->test_limit_arg > config->train_limit_arg &&
           config->train_limit_arg >= 16 &&
           config->test_limit_arg <= (size_t)UINT32_MAX - (size_t)G_MAX &&
           config->train_limit_arg <= (size_t)UINT32_MAX &&
           config->rng_seed_arg <= (size_t)UINT32_MAX &&
           config->regularized_reference_weight >= 0.0 &&
           config->regularized_entropy_weight >= 0.0;
}

static void print_usage(FILE *stream)
{
    fputs("usage: prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] "
          "[max_train_states] [max_test_states] [regularized_reference_weight] "
          "[regularized_entropy_weight] [rng_seed] "
          "[--trace path] [--save path] [--model checkpoint] "
          "[--save-rules path] [--rules path] [--save-learned-model path]\n",
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

static size_t residue_signature(uint32_t p, size_t action, size_t mod_count)
{
    if (action_is_overflow(action)) {
        return (size_t)1u << mod_count;
    }

    uint32_t candidate = p + (uint32_t)action_gap(action);
    size_t signature = 0;
    for (size_t i = 0; i < mod_count; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate > mod && candidate % mod == 0) {
            signature |= (size_t)1u << i;
        }
    }
    return signature;
}

static int feature_wheel_rejects_action(uint32_t p, size_t action)
{
    if (action_is_overflow(action)) {
        return 0;
    }

    uint32_t candidate = p + (uint32_t)action_gap(action);
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate > mod && candidate % mod == 0) {
            return 1;
        }
    }
    return 0;
}

static void apply_feature_wheel_filter(uint32_t p,
                                       nn_real *scores,
                                       nn_real *probabilities)
{
    double survivor_probability = 0.0;
    size_t survivors = 0;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (feature_wheel_rejects_action(p, action)) {
            scores[action] = (nn_real)-1.0e9;
            probabilities[action] = (nn_real)0;
            continue;
        }

        survivor_probability += (double)probabilities[action];
        ++survivors;
    }

    if (survivors == 0 || survivor_probability <= 0.0) {
        nn_real uniform = (nn_real)(1.0 / (double)ACTION_COUNT);
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            scores[action] = uniform;
            probabilities[action] = uniform;
        }
        return;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] =
            (nn_real)((double)probabilities[action] / survivor_probability);
    }
}

#if defined(NN_ENABLE_TEST_API)
static void fill_candidate_features(nn_real *inputs, uint32_t p, double log_scale)
{
    fill_candidate_features_masked(inputs, p, log_scale, &FEATURE_MASK_FULL);
}
#endif

static void rank_actions(const nn_real *scores, size_t *ranked)
{
    nn_rank_action_scores(scores, ACTION_COUNT, ranked);
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

static size_t symbolic_rule_count(void)
{
    size_t count = 0;
    for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
         modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD;
         ++modulus) {
        count += modulus;
    }
    return count;
}

static void fill_symbolic_rule_offsets(size_t *offsets)
{
    size_t offset = 0;
    for (uint32_t modulus = 0; modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD + 1u; ++modulus) {
        offsets[modulus] = 0;
    }
    for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
         modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD;
         ++modulus) {
        offsets[modulus] = offset;
        offset += modulus;
    }
    offsets[(uint32_t)NN_SYMBOLIC_RULE_MAX_MOD + 1u] = offset;
}

static void init_symbolic_rules(SymbolicRejectRule *rules, const size_t *offsets)
{
    for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
         modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD;
         ++modulus) {
        for (uint32_t residue = 0; residue < modulus; ++residue) {
            SymbolicRejectRule *rule = &rules[offsets[modulus] + residue];
            rule->modulus = modulus;
            rule->residue = residue;
        }
    }
}

static void mark_domain_invalid_symbolic_residues(const Sieve *sieve,
                                                  uint8_t *invalid,
                                                  const size_t *offsets)
{
    for (size_t prime_index = 0; prime_index < sieve->prime_count; ++prime_index) {
        uint32_t prime = sieve->primes[prime_index];
        for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
             modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD && modulus < prime;
             ++modulus) {
            invalid[offsets[modulus] + (prime % modulus)] = 1;
        }
    }
}

static void accumulate_symbolic_rule_rejects(SymbolicRejectRule *rules,
                                             const size_t *offsets,
                                             const GapDataset *dataset,
                                             int validation)
{
    for (size_t sample = 0; sample < dataset->count; ++sample) {
        uint32_t p = dataset->prime[sample];
        uint32_t true_gap = dataset->gap[sample];

        for (uint32_t gap = 2; gap < true_gap; gap += 2) {
            uint32_t candidate = p + gap;
            for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
                 modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD && modulus < candidate;
                 ++modulus) {
                SymbolicRejectRule *rule = &rules[offsets[modulus] + (candidate % modulus)];
                if (validation) {
                    ++rule->validation_rejects;
                } else {
                    ++rule->train_rejects;
                }
            }
        }
    }
}

static uint32_t gcd_u32(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

static void fill_symbolic_rule_features(const SymbolicRejectRule *rule,
                                        size_t max_train_rejects,
                                        nn_real *features)
{
    double modulus = (double)rule->modulus;
    double residue = (double)rule->residue;
    double max_modulus = (double)NN_SYMBOLIC_RULE_MAX_MOD;
    double support_scale = max_train_rejects == 0 ? 1.0 : (double)max_train_rejects;
    uint32_t common = gcd_u32(rule->modulus, rule->residue);

    features[0] = (nn_real)(modulus / max_modulus);
    features[1] = (nn_real)(residue / modulus);
    features[2] = rule->residue == 0 ? (nn_real)1 : (nn_real)0;
    features[3] = (nn_real)((double)common / modulus);
    features[4] = (nn_real)((double)rule->train_rejects / support_scale);
}

static int score_symbolic_rules_with_model(SymbolicRejectRule *rules,
                                           size_t count,
                                           double *loss_out)
{
    size_t max_train_rejects = 0;
    for (size_t i = 0; i < count; ++i) {
        if (rules[i].train_rejects > max_train_rejects) {
            max_train_rejects = rules[i].train_rejects;
        }
    }

    nn_real *inputs = calloc(count * SYMBOLIC_RULE_FEATURE_COUNT, sizeof(*inputs));
    nn_real *targets = calloc(count, sizeof(*targets));
    nn_real *scores = calloc(count, sizeof(*scores));
    if (inputs == NULL || targets == NULL || scores == NULL) {
        free(inputs);
        free(targets);
        free(scores);
        return 0;
    }

    double support_scale = max_train_rejects == 0 ? 1.0 : (double)max_train_rejects;
    for (size_t i = 0; i < count; ++i) {
        fill_symbolic_rule_features(&rules[i],
                                    max_train_rejects,
                                    &inputs[i * SYMBOLIC_RULE_FEATURE_COUNT]);
        targets[i] = rules[i].domain_valid
                         ? (nn_real)((double)rules[i].train_rejects / support_scale)
                         : (nn_real)0;
    }

    const size_t layers[] = {SYMBOLIC_RULE_FEATURE_COUNT, 16, 8, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 7319;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        free(inputs);
        free(targets);
        free(scores);
        return 0;
    }

    nn_real loss = nn_train_batched(net,
                                    inputs,
                                    targets,
                                    count,
                                    256,
                                    (nn_real)0.02,
                                    SYMBOLIC_RULE_SCORER_EPOCHS,
                                    1);
    nn_forward_batch(net, inputs, scores, count);
    for (size_t i = 0; i < count; ++i) {
        rules[i].model_score = scores[i];
    }
    if (loss_out != NULL) {
        *loss_out = (double)loss;
    }

    nn_free(net);
    free(inputs);
    free(targets);
    free(scores);
    return 1;
}

static int compare_symbolic_rules(const void *left, const void *right)
{
    const SymbolicRejectRule *a = (const SymbolicRejectRule *)left;
    const SymbolicRejectRule *b = (const SymbolicRejectRule *)right;

    if ((double)a->model_score < (double)b->model_score) {
        return 1;
    }
    if ((double)a->model_score > (double)b->model_score) {
        return -1;
    }
    if (a->train_rejects != b->train_rejects) {
        return a->train_rejects < b->train_rejects ? 1 : -1;
    }
    if (a->validation_rejects != b->validation_rejects) {
        return a->validation_rejects < b->validation_rejects ? 1 : -1;
    }
    if (a->modulus != b->modulus) {
        return a->modulus < b->modulus ? -1 : 1;
    }
    if (a->residue != b->residue) {
        return a->residue < b->residue ? -1 : 1;
    }
    return 0;
}

static int compare_symbolic_rules_canonical(const void *left, const void *right)
{
    const SymbolicRejectRule *a = (const SymbolicRejectRule *)left;
    const SymbolicRejectRule *b = (const SymbolicRejectRule *)right;

    if (a->modulus != b->modulus) {
        return a->modulus < b->modulus ? -1 : 1;
    }
    if (a->residue != b->residue) {
        return a->residue < b->residue ? -1 : 1;
    }
    return 0;
}


static int symbolic_rule_subset_of(const SymbolicRejectRule *candidate,
                                   const SymbolicRejectRule *accepted)
{
    return candidate->modulus % accepted->modulus == 0 &&
           candidate->residue % accepted->modulus == accepted->residue;
}

static int symbolic_rule_is_covered(const SymbolicRuleSet *accepted,
                                    const SymbolicRejectRule *candidate)
{
    for (size_t i = 0; i < accepted->count; ++i) {
        if (symbolic_rule_subset_of(candidate, &accepted->items[i])) {
            return 1;
        }
    }
    return 0;
}

static void symbolic_rule_remove_subsets(SymbolicRuleSet *accepted,
                                         const SymbolicRejectRule *candidate)
{
    size_t write = 0;
    for (size_t read = 0; read < accepted->count; ++read) {
        if (symbolic_rule_subset_of(&accepted->items[read], candidate)) {
            continue;
        }
        if (write != read) {
            accepted->items[write] = accepted->items[read];
        }
        ++write;
    }
    accepted->count = write;
}

static void free_symbolic_rule_set(SymbolicRuleSet *rules)
{
    free(rules->items);
    memset(rules, 0, sizeof(*rules));
}

static int save_symbolic_rule_checkpoint(const SymbolicRuleSet *rules,
                                         const char *path,
                                         const SymbolicWheelMetrics *metrics,
                                         char out_hash[65])
{
    if (rules == NULL || path == NULL) {
        return 0;
    }

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    int ok = fprintf(fp, "crlp.symbolic_rules.v1\n") >= 0 &&
             fprintf(fp, "rule_count %zu\n", rules->count) >= 0 &&
             fprintf(fp, "max_modulus %u\n", (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD) >= 0 &&
             fprintf(fp, "feature_mod_count %u\n", (uint32_t)FEATURE_MOD_COUNT) >= 0;
    if (metrics != NULL) {
        ok = ok &&
             fprintf(fp,
                     "proposed_rules %zu\n"
                     "domain_valid_rules %zu\n"
                     "accepted_train_reject_support %zu\n"
                     "validation_reject_support %zu\n"
                     "proposer_loss %.17g\n",
                     metrics->proposed_rules,
                     metrics->domain_valid_rules,
                     metrics->train_rejects,
                     metrics->validation_rejects,
                     metrics->proposer_loss) >= 0;
    } else {
        ok = ok &&
             fprintf(fp,
                     "proposed_rules 0\n"
                     "domain_valid_rules 0\n"
                     "accepted_train_reject_support 0\n"
                     "validation_reject_support 0\n"
                     "proposer_loss 0\n") >= 0;
    }

    for (size_t i = 0; ok && i < rules->count; ++i) {
        const SymbolicRejectRule *rule = &rules->items[i];
        ok = fprintf(fp,
                     "rule %u %u %zu %zu %.9g\n",
                     rule->modulus,
                     rule->residue,
                     rule->train_rejects,
                     rule->validation_rejects,
                     (double)rule->model_score) >= 0;
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok) {
        return 0;
    }
    if (out_hash != NULL && !nn_hash_checkpoint_hex(path, out_hash)) {
        return 0;
    }
    return 1;
}

static int load_symbolic_rule_checkpoint(const char *path, SymbolicRuleSet *rules)
{
    memset(rules, 0, sizeof(*rules));
    if (path == NULL) {
        return 0;
    }

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    char magic[64];
    size_t count = 0;
    unsigned int max_modulus = 0;
    unsigned int feature_mod_count = 0;
    int ok = fscanf(fp, "%63s", magic) == 1 &&
             strcmp(magic, "crlp.symbolic_rules.v1") == 0 &&
             fscanf(fp, " rule_count %zu", &count) == 1 &&
             fscanf(fp, " max_modulus %u", &max_modulus) == 1 &&
             fscanf(fp, " feature_mod_count %u", &feature_mod_count) == 1;

    size_t ignored_size = 0;
    double ignored_loss = 0.0;
    ok = ok &&
         fscanf(fp, " proposed_rules %zu", &ignored_size) == 1 &&
         fscanf(fp, " domain_valid_rules %zu", &ignored_size) == 1 &&
         fscanf(fp, " accepted_train_reject_support %zu", &ignored_size) == 1 &&
         fscanf(fp, " validation_reject_support %zu", &ignored_size) == 1 &&
         fscanf(fp, " proposer_loss %lf", &ignored_loss) == 1;
    ok = ok && max_modulus == (unsigned int)NN_SYMBOLIC_RULE_MAX_MOD &&
         feature_mod_count == (unsigned int)FEATURE_MOD_COUNT;

    if (ok) {
        rules->items = calloc(count == 0 ? 1 : count, sizeof(*rules->items));
        rules->capacity = count;
        ok = rules->items != NULL;
    }

    for (size_t i = 0; ok && i < count; ++i) {
        unsigned int modulus = 0;
        unsigned int residue = 0;
        size_t train_rejects = 0;
        size_t validation_rejects = 0;
        double model_score = 0.0;
        ok = fscanf(fp,
                    " rule %u %u %zu %zu %lf",
                    &modulus,
                    &residue,
                    &train_rejects,
                    &validation_rejects,
                    &model_score) == 5 &&
             modulus >= SYMBOLIC_RULE_MOD_MIN &&
             modulus <= max_modulus &&
             residue < modulus;
        if (!ok) {
            break;
        }

        rules->items[i].modulus = (uint32_t)modulus;
        rules->items[i].residue = (uint32_t)residue;
        rules->items[i].train_rejects = train_rejects;
        rules->items[i].validation_rejects = validation_rejects;
        rules->items[i].model_score = (nn_real)model_score;
        rules->items[i].domain_valid = 1;
        ++rules->count;
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || rules->count != count) {
        free_symbolic_rule_set(rules);
        return 0;
    }

    qsort(rules->items,
          rules->count,
          sizeof(*rules->items),
          compare_symbolic_rules_canonical);
    return 1;
}

static int load_symbolic_rules_with_hash(const char *path,
                                         SymbolicRuleSet *rules,
                                         char out_hash[65])
{
    if (!load_symbolic_rule_checkpoint(path, rules)) {
        return 0;
    }
    if (out_hash != NULL && !nn_hash_checkpoint_hex(path, out_hash)) {
        strcpy(out_hash, "unknown");
    }
    return 1;
}

static int discover_symbolic_wheel(const Sieve *sieve,
                                   const GapDataset *train,
                                   const GapDataset *validation,
                                   SymbolicRuleSet *accepted,
                                   SymbolicWheelMetrics *metrics)
{
    size_t offsets[(uint32_t)NN_SYMBOLIC_RULE_MAX_MOD + 2u];
    size_t count = symbolic_rule_count();
    SymbolicRejectRule *rules = calloc(count, sizeof(*rules));
    uint8_t *invalid = calloc(count, sizeof(*invalid));

    memset(accepted, 0, sizeof(*accepted));
    memset(metrics, 0, sizeof(*metrics));
    metrics->proposed_rules = count;

    if (rules == NULL || invalid == NULL) {
        free(rules);
        free(invalid);
        return 0;
    }

    fill_symbolic_rule_offsets(offsets);
    init_symbolic_rules(rules, offsets);
    mark_domain_invalid_symbolic_residues(sieve, invalid, offsets);

    for (size_t i = 0; i < count; ++i) {
        rules[i].domain_valid = invalid[i] ? 0 : 1;
        metrics->domain_valid_rules += rules[i].domain_valid ? 1u : 0u;
    }

    accumulate_symbolic_rule_rejects(rules, offsets, train, 0);
    accumulate_symbolic_rule_rejects(rules, offsets, validation, 1);
    if (!score_symbolic_rules_with_model(rules, count, &metrics->proposer_loss)) {
        free_symbolic_rule_set(accepted);
        free(rules);
        free(invalid);
        return 0;
    }
    qsort(rules, count, sizeof(*rules), compare_symbolic_rules);

    accepted->capacity = NN_SYMBOLIC_MAX_LEARNED_RULES;
    accepted->items = calloc(accepted->capacity, sizeof(*accepted->items));
    if (accepted->items == NULL) {
        free(rules);
        free(invalid);
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        if (!rules[i].domain_valid || rules[i].train_rejects == 0) {
            continue;
        }
        if (symbolic_rule_is_covered(accepted, &rules[i])) {
            continue;
        }

        symbolic_rule_remove_subsets(accepted, &rules[i]);
        if (accepted->count >= accepted->capacity) {
            continue;
        }
        accepted->items[accepted->count++] = rules[i];
    }

    metrics->accepted_rules = accepted->count;
    qsort(accepted->items,
          accepted->count,
          sizeof(*accepted->items),
          compare_symbolic_rules_canonical);
    for (size_t i = 0; i < accepted->count; ++i) {
        metrics->train_rejects += accepted->items[i].train_rejects;
        metrics->validation_rejects += accepted->items[i].validation_rejects;
    }
    free(rules);
    free(invalid);
    return 1;
}

static int symbolic_rule_rejects(const SymbolicRejectRule *rule, uint32_t candidate)
{
    return candidate > rule->modulus && candidate % rule->modulus == rule->residue;
}

static int learned_symbolic_probe_gap(const Sieve *sieve,
                                      const SymbolicRuleSet *rules,
                                      uint32_t p,
                                      uint32_t gap,
                                      VerifierWork *work,
                                      int *rejected)
{
    uint32_t candidate = p + gap;

    ++work->candidates;
    *rejected = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        ++work->mod_tests;
        if (symbolic_rule_rejects(&rules->items[i], candidate)) {
            *rejected = 1;
            return 0;
        }
    }

    ++work->exact_tests;
    return is_prime_sieve(sieve, candidate);
}

static VerifierWork learned_symbolic_scan_work(const Sieve *sieve,
                                               const SymbolicRuleSet *rules,
                                               uint32_t p,
                                               uint32_t true_gap,
                                               int *false_rejection)
{
    VerifierWork work = {0, 0, 0};
    *false_rejection = 0;

    for (uint32_t gap = 2; gap <= true_gap; gap += 2) {
        int rejected = 0;
        int prime = learned_symbolic_probe_gap(sieve, rules, p, gap, &work, &rejected);
        if (rejected && gap == true_gap) {
            *false_rejection = 1;
            return work;
        }
        if (prime) {
            return work;
        }
    }

    return work;
}

static SymbolicWheelMetrics evaluate_symbolic_wheel(const Sieve *sieve,
                                                    const SymbolicRuleSet *rules,
                                                    const GapDataset *dataset,
                                                    size_t max_states,
                                                    SymbolicWheelMetrics metrics)
{
    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];
        VerifierWork direct = direct_scan_work(gap);
        VerifierWork hand = wheel_scan_work(sieve, p, gap);
        int false_rejection = 0;
        VerifierWork learned = learned_symbolic_scan_work(sieve,
                                                          rules,
                                                          p,
                                                          gap,
                                                          &false_rejection);

        metrics.direct_exact_tests += (double)direct.exact_tests;
        metrics.hand_wheel_mod_tests += (double)hand.mod_tests;
        metrics.hand_wheel_exact_tests += (double)hand.exact_tests;
        metrics.learned_mod_tests += (double)learned.mod_tests;
        metrics.learned_exact_tests += (double)learned.exact_tests;
        metrics.test_false_rejections += false_rejection ? 1u : 0u;
    }

    metrics.samples = samples;
    if (samples != 0) {
        double denominator = (double)samples;
        metrics.direct_exact_tests /= denominator;
        metrics.hand_wheel_mod_tests /= denominator;
        metrics.hand_wheel_exact_tests /= denominator;
        metrics.learned_mod_tests /= denominator;
        metrics.learned_exact_tests /= denominator;
    }
    return metrics;
}

static void print_symbolic_rule(const SymbolicRejectRule *rule)
{
    printf("n %% %u == %u", rule->modulus, rule->residue);
}

static void print_symbolic_wheel_report(const SymbolicRuleSet *rules,
                                        SymbolicWheelMetrics metrics)
{
    double learned_saved = metrics.direct_exact_tests - metrics.learned_exact_tests;
    double hand_saved = metrics.direct_exact_tests - metrics.hand_wheel_exact_tests;
    double learned_saved_pct = metrics.direct_exact_tests == 0.0
                                   ? 0.0
                                   : 100.0 * learned_saved / metrics.direct_exact_tests;
    double hand_saved_pct = metrics.direct_exact_tests == 0.0
                                ? 0.0
                                : 100.0 * hand_saved / metrics.direct_exact_tests;

    printf("\n[learned symbolic wheel]\n");
    printf("  grammar: reject candidate n when n > m and n %% m == r; m=3..%u\n",
           (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD);
    printf("  proposer/verifier\n");
    printf("    neural_rule_scorer_epochs=%u final_mse=%.6f\n",
           (uint32_t)SYMBOLIC_RULE_SCORER_EPOCHS,
           metrics.proposer_loss);
    printf("    proposed_rules=%zu domain_valid=%zu accepted=%zu max_rules=%u\n",
           metrics.proposed_rules,
           metrics.domain_valid_rules,
           metrics.accepted_rules,
           (uint32_t)NN_SYMBOLIC_MAX_LEARNED_RULES);
    printf("    accepted_train_reject_support=%zu validation_reject_support=%zu\n",
           metrics.train_rejects,
           metrics.validation_rejects);
    printf("  accepted rules");
    for (size_t i = 0; i < rules->count; ++i) {
        printf("%s", i == 0 ? ": " : ", ");
        print_symbolic_rule(&rules->items[i]);
        if (i == 15 && rules->count > 16) {
            printf(", ...");
            break;
        }
    }
    if (rules->count == 0) {
        printf(": none");
    }
    printf("\n");
    printf("  verifier diagnostics\n");
    printf("    samples=%zu false_rejections=%zu\n",
           metrics.samples,
           metrics.test_false_rejections);
    printf("    direct exact=%.2f\n", metrics.direct_exact_tests);
    printf("    hand wheel exact=%.2f saved=%.2f (%.1f%%) mod_tests=%.2f\n",
           metrics.hand_wheel_exact_tests,
           hand_saved,
           hand_saved_pct,
           metrics.hand_wheel_mod_tests);
    printf("    learned wheel exact=%.2f saved=%.2f (%.1f%%) mod_tests=%.2f\n",
           metrics.learned_exact_tests,
           learned_saved,
           learned_saved_pct,
           metrics.learned_mod_tests);
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

static size_t fill_symbolic_survivor_candidates(const SymbolicRuleSet *rules,
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

static size_t survivor_index_for_action(const size_t *survivor_actions,
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

static EvalMetrics evaluate_model_symbolic_masked(NeuralNet *net,
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

static double fit_temperature_symbolic_masked(NeuralNet *net,
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

static EvalMetrics evaluate_frequency_policy(const Sieve *sieve,
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

static void print_metrics(const char *label, EvalMetrics metrics)
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

static int prime_action_from_action_id(uint32_t action_id, size_t *out_action)
{
    if (action_id >= ACTION_COUNT || out_action == NULL) {
        return 0;
    }

    *out_action = (size_t)action_id;
    return 1;
}

static void fill_action_ids(uint32_t *action_ids)
{
    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        action_ids[action] = prime_action_id_for_action(action);
    }
}

static size_t action_row_for_action_id(const uint32_t *action_ids,
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

static int save_prime_checkpoint_with_meta(NeuralNet *net,
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

static int save_prime_checkpoint(NeuralNet *net, const char *path, char out_hash[65])
{
    return save_prime_checkpoint_with_meta(net,
                                           path,
                                           "prime_gap full-feature action scorer",
                                           "unknown",
                                           NULL,
                                           out_hash);
}

static int emit_prime_gap_trace(NeuralNet *net,
                                const Sieve *sieve,
                                const GapDataset *dataset,
                                size_t max_states,
                                double log_scale,
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
        fill_candidate_features_masked(candidate_inputs, p, log_scale, &FEATURE_MASK_FULL);
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
                 "\"rule_set_hash\":\"%s\"",
                 p,
                 gap,
                 proposal_steps,
                 wheel.exact_tests,
                 assisted.exact_tests,
                 block_work_cost(block),
                 block_work_cost(assisted_block),
                 log_scale,
                 rule_hash == NULL ? "unknown" : rule_hash);

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

static void free_conditioned_frequency_policy(ConditionedFrequencyPolicy *policy)
{
    if (policy == NULL) {
        return;
    }

    free(policy->action_counts);
    memset(policy, 0, sizeof(*policy));
}

static int build_conditioned_frequency_policy(ConditionedFrequencyPolicy *policy,
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

static void conditioned_frequency_scores(const ConditionedFrequencyPolicy *policy,
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

static EvalMetrics evaluate_conditioned_frequency_policy(
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
            nn_real step_loss = nn_train_action_scorer(net,
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

static int train_pairwise_ranker(NeuralNet *net,
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
            nn_real step_loss = nn_train_pairwise_action_ranker(net,
                                                                candidate_inputs,
                                                                ACTION_COUNT,
                                                                gap_to_action(dataset->gap[index]),
                                                                (nn_real)1,
                                                                learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu pairwise_hinge=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               loss / (double)train_steps,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

static int train_supervised_symbolic_masked(NeuralNet *net,
                                            const char *label,
                                            const SymbolicRuleSet *rules,
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
    size_t survivor_actions[ACTION_COUNT];
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double loss = 0.0;
        size_t masked_out_targets = 0;
        size_t survivor_total = 0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            size_t target = gap_to_action(dataset->gap[index]);
            size_t survivor_count = fill_symbolic_survivor_candidates(rules,
                                                                      candidate_inputs,
                                                                      dataset->prime[index],
                                                                      log_scale,
                                                                      mask,
                                                                      survivor_actions);
            size_t survivor_target =
                survivor_index_for_action(survivor_actions, survivor_count, target);
            if (survivor_count == 0 || survivor_target == survivor_count) {
                ++masked_out_targets;
                continue;
            }
            survivor_total += survivor_count;

            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       survivor_count,
                                                       survivor_target,
                                                       learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        double divisor = train_steps == masked_out_targets
                             ? 1.0
                             : (double)(train_steps - masked_out_targets);
        printf("%s epoch %zu loss=%.4f updates=%zu masked_targets=%zu "
               "mean_survivors=%.2f %.0f states/s\n",
               label,
               epoch + 1,
               loss / divisor,
               train_steps - masked_out_targets,
               masked_out_targets,
               survivor_total / divisor,
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
            if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, probabilities)) {
                return 0;
            }

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
            if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, probabilities)) {
                return 0;
            }
            if (!nn_score_action_rows(reference_net,
                                      candidate_inputs,
                                      ACTION_COUNT,
                                      reference_scores,
                                      reference_probabilities)) {
                return 0;
            }

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

static int run_linear_scorer_baseline(const NNConfig *config,
                                      const Sieve *sieve,
                                      const GapDataset *train,
                                      const GapDataset *validation,
                                      const GapDataset *test,
                                      size_t max_train_states,
                                      size_t max_validation_states,
                                      size_t max_test_states,
                                      size_t supervised_epochs,
                                      double log_scale,
                                      nn_real *candidate_inputs,
                                      nn_real *scores,
                                      nn_real *probabilities,
                                      size_t *order,
                                      uint32_t *rng,
                                      EvalMetrics *metrics_out,
                                      EvalMetrics *temp_metrics_out,
                                      double *temperature_out)
{
    NeuralNet *net = nn_create_config(config);
    int ok;

    if (net == NULL) {
        return 0;
    }

    printf("\ntraining baseline linear-scorer (full features, no hidden layer)\n");
    ok = train_supervised(net,
                          "baseline linear-scorer",
                          train,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          candidate_inputs,
                          order,
                          rng);
    if (ok) {
        double validation_nll = 0.0;
        double temperature = fit_temperature(net,
                                             validation,
                                             max_validation_states,
                                             log_scale,
                                             &FEATURE_MASK_FULL,
                                             candidate_inputs,
                                             scores,
                                             probabilities,
                                             &validation_nll);
        printf("baseline linear-scorer temperature=%.2f validation_nll=%.4f\n",
               temperature,
               validation_nll);

        *metrics_out = evaluate_model(net,
                                      sieve,
                                      test,
                                      max_test_states,
                                      log_scale,
                                      &FEATURE_MASK_FULL,
                                      0,
                                      1.0,
                                      candidate_inputs,
                                      scores,
                                      probabilities);
        *temp_metrics_out = evaluate_model(net,
                                           sieve,
                                           test,
                                           max_test_states,
                                           log_scale,
                                           &FEATURE_MASK_FULL,
                                           0,
                                           temperature,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
        *temperature_out = temperature;
        print_metrics("baseline linear-scorer", *metrics_out);
        print_metrics("baseline linear-scorer temp-scaled", *temp_metrics_out);
    }

    nn_free(net);
    return ok;
}

int nn_prime_gap_main(int argc, char **argv)
{
    PrimeGapRunConfig run_config;
    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    uint32_t rng;
    int ok = 1;
    char model_hash[65] = "unknown";
    char rule_hash[65] = "unknown";
    int rules_loaded_from_checkpoint = 0;

    if (!parse_run_config(argc, argv, &run_config) || !run_config_is_valid(&run_config)) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }

    rng = (uint32_t)run_config.rng_seed_arg;

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
    printf("regularized_rl reference_weight=%.3f entropy_weight=%.3f rng_seed=%u\n",
           run_config.regularized_reference_weight,
           run_config.regularized_entropy_weight,
           rng);

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

    const size_t layers[] = {FEATURE_COUNT, NN_PRIME_GAP_HIDDEN1, NN_PRIME_GAP_HIDDEN2, 1};
    const size_t linear_layers[] = {FEATURE_COUNT, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 42;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    NNConfig linear_config = config;
    linear_config.layer_sizes = linear_layers;
    linear_config.layer_count = sizeof(linear_layers) / sizeof(linear_layers[0]);
    linear_config.hidden_activation = NN_ACTIVATION_LINEAR;
    linear_config.seed = 4201;

    NeuralNet *net = nn_create_config(&config);
    NeuralNet *learned_wheel_net = NULL;
    NeuralNet *pairwise_net = NULL;
    NeuralNet *strict_net = NULL;
    NeuralNet *regularized_net = NULL;
    ConditionedFrequencyPolicy conditioned_policy = {0};
    SymbolicRuleSet learned_symbolic_rules = {0};
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

    if (run_config.load_rules_path != NULL) {
        if (!load_symbolic_rules_with_hash(run_config.load_rules_path,
                                           &learned_symbolic_rules,
                                           rule_hash)) {
            fprintf(stderr, "failed to load symbolic rules: %s\n", run_config.load_rules_path);
            ok = 0;
            goto cleanup;
        }
        rules_loaded_from_checkpoint = 1;
        printf("loaded symbolic rules: %s hash=%s rules=%zu\n",
               run_config.load_rules_path,
               rule_hash,
               learned_symbolic_rules.count);
    }

    if (run_config.load_model_path != NULL) {
        NNCheckpointMeta loaded_meta;
        NeuralNet *loaded_net = nn_load_checkpoint(run_config.load_model_path, &loaded_meta);
        if (loaded_net == NULL) {
            fprintf(stderr, "failed to load checkpoint: %s\n", run_config.load_model_path);
            ok = 0;
            goto cleanup;
        }

        nn_free(net);
        net = loaded_net;
        if (!nn_hash_checkpoint_hex(run_config.load_model_path, model_hash)) {
            strcpy(model_hash, "unknown");
        }
        printf("loaded checkpoint: %s hash=%s feature_encoder_version=%u parent=%s\n",
               run_config.load_model_path,
               model_hash,
               loaded_meta.feature_encoder_version,
               loaded_meta.parent_model_hash[0] == '\0' ? "unknown"
                                                         : loaded_meta.parent_model_hash);
        print_metrics("loaded model",
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
        if (rules_loaded_from_checkpoint) {
            print_metrics("loaded model learned-wheel",
                          evaluate_model_symbolic_masked(net,
                                                         &sieve,
                                                         &learned_symbolic_rules,
                                                         &test,
                                                         run_config.max_test_states,
                                                         log_scale,
                                                         &FEATURE_MASK_FULL,
                                                         1,
                                                         1.0,
                                                         candidate_inputs,
                                                         scores,
                                                         probabilities));
        }
        if (run_config.trace_path != NULL &&
            !emit_prime_gap_trace(net,
                                  &sieve,
                                  &test,
                                  run_config.max_test_states,
                                  log_scale,
                                  run_config.trace_path,
                                  model_hash,
                                  rule_hash,
                                  (uint64_t)rng,
                                  candidate_inputs,
                                  scores,
                                  probabilities)) {
            fputs("failed to write trace\n", stderr);
            ok = 0;
        }
        goto cleanup;
    }

    size_t frequency_ranked[ACTION_COUNT];
    nn_real frequency_probabilities[ACTION_COUNT];
    EvalMetrics frequency_metrics;
    EvalMetrics wheel_frequency_metrics;
    EvalMetrics conditioned_metrics;
    EvalMetrics wheel_conditioned_metrics;
    EvalMetrics model_before_metrics;
    EvalMetrics linear_metrics;
    EvalMetrics linear_temp_metrics;
    EvalMetrics learned_wheel_metrics;
    EvalMetrics learned_wheel_temp_metrics;
    EvalMetrics pairwise_metrics;
    double linear_temperature = 1.0;
    double learned_wheel_temperature = 1.0;

    printf("\n=== Baselines ===\n");
    build_frequency_policy(&train, frequency_ranked, frequency_probabilities);
    (void)frequency_ranked;
    frequency_metrics = evaluate_frequency_policy(&sieve,
                                                  &test,
                                                  frequency_probabilities,
                                                  run_config.max_test_states,
                                                  1,
                                                  0);
    print_metrics("baseline empirical-gap", frequency_metrics);
    wheel_frequency_metrics = evaluate_frequency_policy(&sieve,
                                                        &test,
                                                        frequency_probabilities,
                                                        run_config.max_test_states,
                                                        1,
                                                        1);
    print_metrics("baseline hand-wheel empirical", wheel_frequency_metrics);

    if (!build_conditioned_frequency_policy(&conditioned_policy, &train)) {
        fputs("failed to build conditioned frequency policy\n", stderr);
        ok = 0;
        goto cleanup;
    }
    conditioned_metrics = evaluate_conditioned_frequency_policy(&conditioned_policy,
                                                                &sieve,
                                                                &test,
                                                                run_config.max_test_states,
                                                                1,
                                                                0,
                                                                scores,
                                                                probabilities);
    print_metrics("baseline residue-conditioned", conditioned_metrics);
    wheel_conditioned_metrics = evaluate_conditioned_frequency_policy(&conditioned_policy,
                                                                      &sieve,
                                                                      &test,
                                                                      run_config.max_test_states,
                                                                      1,
                                                                      1,
                                                                      scores,
                                                                      probabilities);
    print_metrics("baseline hand-wheel residue-conditioned", wheel_conditioned_metrics);
    free_conditioned_frequency_policy(&conditioned_policy);

    SymbolicWheelMetrics symbolic_metrics;
    if (rules_loaded_from_checkpoint) {
        memset(&symbolic_metrics, 0, sizeof(symbolic_metrics));
        symbolic_metrics.accepted_rules = learned_symbolic_rules.count;
    } else {
        if (!discover_symbolic_wheel(&sieve,
                                     &train,
                                     &validation,
                                     &learned_symbolic_rules,
                                     &symbolic_metrics)) {
            fputs("failed to discover symbolic wheel\n", stderr);
            ok = 0;
            goto cleanup;
        }
    }
    symbolic_metrics = evaluate_symbolic_wheel(&sieve,
                                               &learned_symbolic_rules,
                                               &test,
                                               run_config.max_test_states,
                                               symbolic_metrics);
    print_symbolic_wheel_report(&learned_symbolic_rules, symbolic_metrics);

    if (run_config.save_rules_path != NULL &&
        !save_symbolic_rule_checkpoint(&learned_symbolic_rules,
                                       run_config.save_rules_path,
                                       &symbolic_metrics,
                                       rule_hash)) {
        fputs("failed to save symbolic rules\n", stderr);
        ok = 0;
        goto cleanup;
    }
    if (run_config.save_rules_path != NULL) {
        printf("saved symbolic rules: %s hash=%s rules=%zu\n",
               run_config.save_rules_path,
               rule_hash,
               learned_symbolic_rules.count);
    }

    uint32_t linear_rng = rng;
    fill_order(order, train.count);
    ok = run_linear_scorer_baseline(&linear_config,
                                    &sieve,
                                    &train,
                                    &validation,
                                    &test,
                                    run_config.max_train_states,
                                    max_validation_states,
                                    run_config.max_test_states,
                                    run_config.supervised_epochs,
                                    log_scale,
                                    candidate_inputs,
                                    scores,
                                    probabilities,
                                    order,
                                    &linear_rng,
                                    &linear_metrics,
                                    &linear_temp_metrics,
                                    &linear_temperature);
    if (!ok) {
        goto cleanup;
    }
    fill_order(order, train.count);

    model_before_metrics = evaluate_model(net,
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
    print_metrics("model before", model_before_metrics);

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

    printf("\n=== Ranking Loss Ablation ===\n");
    pairwise_net = nn_create_config(&config);
    if (pairwise_net == NULL) {
        fputs("failed to allocate pairwise-ranking model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    fill_order(order, train.count);
    ok = train_pairwise_ranker(pairwise_net,
                               "model pairwise-ranking",
                               &train,
                               run_config.max_train_states,
                               run_config.supervised_epochs,
                               (nn_real)0.002,
                               log_scale,
                               &FEATURE_MASK_FULL,
                               candidate_inputs,
                               order,
                               &rng);
    if (!ok) {
        goto cleanup;
    }
    pairwise_metrics = evaluate_model(pairwise_net,
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
    print_metrics("model pairwise-ranking", pairwise_metrics);

    printf("\n=== Learned-Wheel Proposal Training ===\n");
    learned_wheel_net = nn_create_config(&config);
    if (learned_wheel_net == NULL) {
        fputs("failed to allocate learned-wheel model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    fill_order(order, train.count);
    ok = train_supervised_symbolic_masked(learned_wheel_net,
                                          "model learned-wheel supervised",
                                          &learned_symbolic_rules,
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

    double learned_wheel_temperature_nll = 0.0;
    learned_wheel_temperature = fit_temperature_symbolic_masked(learned_wheel_net,
                                                                &learned_symbolic_rules,
                                                                &validation,
                                                                max_validation_states,
                                                                log_scale,
                                                                &FEATURE_MASK_FULL,
                                                                candidate_inputs,
                                                                scores,
                                                                probabilities,
                                                                &learned_wheel_temperature_nll);
    printf("model learned-wheel supervised temperature=%.2f validation_nll=%.4f\n",
           learned_wheel_temperature,
           learned_wheel_temperature_nll);

    learned_wheel_metrics = evaluate_model_symbolic_masked(learned_wheel_net,
                                                           &sieve,
                                                           &learned_symbolic_rules,
                                                           &test,
                                                           run_config.max_test_states,
                                                           log_scale,
                                                           &FEATURE_MASK_FULL,
                                                           1,
                                                           1.0,
                                                           candidate_inputs,
                                                           scores,
                                                           probabilities);
    learned_wheel_temp_metrics = evaluate_model_symbolic_masked(learned_wheel_net,
                                                                &sieve,
                                                                &learned_symbolic_rules,
                                                                &test,
                                                                run_config.max_test_states,
                                                                log_scale,
                                                                &FEATURE_MASK_FULL,
                                                                0,
                                                                learned_wheel_temperature,
                                                                candidate_inputs,
                                                                scores,
                                                                probabilities);
    print_metrics("model learned-wheel supervised", learned_wheel_metrics);
    print_metrics("model learned-wheel supervised temp-scaled",
                  learned_wheel_temp_metrics);

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
    print_comparison_row("empirical-gap", 1.0, frequency_metrics);
    print_comparison_row("hand-wheel empirical", 1.0, wheel_frequency_metrics);
    print_comparison_row("residue-conditioned", 1.0, conditioned_metrics);
    print_comparison_row("hand-wheel residue-conditioned", 1.0, wheel_conditioned_metrics);
    print_comparison_row("linear-scorer", 1.0, linear_metrics);
    print_comparison_row("linear-scorer temp-scaled", linear_temperature, linear_temp_metrics);
    print_comparison_row("supervised", 1.0, supervised_metrics);
    print_comparison_row("supervised temp-scaled",
                         supervised_temperature,
                         supervised_temp_metrics);
    print_comparison_row("pairwise-ranking", 1.0, pairwise_metrics);
    print_comparison_row("learned-wheel supervised", 1.0, learned_wheel_metrics);
    print_comparison_row("learned-wheel supervised temp-scaled",
                         learned_wheel_temperature,
                         learned_wheel_temp_metrics);
    if (have_rl_metrics) {
        print_comparison_row("strict-rl", 1.0, strict_metrics);
        print_comparison_row("strict-rl temp-scaled", rl_temperature, strict_temp_metrics);
        print_comparison_row("regularized-rl", 1.0, regularized_metrics);
        print_comparison_row("regularized-rl temp-scaled",
                             regularized_temperature,
                             regularized_temp_metrics);
    }

    if (run_config.save_path != NULL &&
        !save_prime_checkpoint(net, run_config.save_path, model_hash)) {
        fputs("failed to save checkpoint\n", stderr);
        ok = 0;
        goto cleanup;
    }

    if (run_config.save_learned_model_path != NULL) {
        char learned_model_hash[65] = "unknown";
        if (!save_prime_checkpoint_with_meta(learned_wheel_net,
                                             run_config.save_learned_model_path,
                                             "prime_gap learned-wheel action scorer",
                                             "unknown",
                                             rule_hash,
                                             learned_model_hash)) {
            fputs("failed to save learned-wheel checkpoint\n", stderr);
            ok = 0;
            goto cleanup;
        }
    }

    if (run_config.trace_path != NULL &&
        !emit_prime_gap_trace(net,
                              &sieve,
                              &test,
                              run_config.max_test_states,
                              log_scale,
                              run_config.trace_path,
                              model_hash,
                              rule_hash,
                              (uint64_t)run_config.rng_seed_arg,
                              candidate_inputs,
                              scores,
                              probabilities)) {
        fputs("failed to write trace\n", stderr);
        ok = 0;
        goto cleanup;
    }

cleanup:
    free_symbolic_rule_set(&learned_symbolic_rules);
    free_conditioned_frequency_policy(&conditioned_policy);
    free(order);
    free(reference_probabilities);
    free(reference_scores);
    free(probabilities);
    free(scores);
    free(candidate_inputs);
    nn_free(regularized_net);
    nn_free(strict_net);
    nn_free(pairwise_net);
    nn_free(learned_wheel_net);
    nn_free(net);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

typedef struct {
    uint32_t p;
    uint32_t target_action_id;
    double log_scale;
} ReplaySample;

typedef struct {
    ReplaySample *items;
    size_t count;
    size_t capacity;
    char parent_hash[65];
    char rule_hash[65];
} ReplayTrace;

static int append_replay_sample(ReplayTrace *trace, ReplaySample sample)
{
    if (trace->count == trace->capacity) {
        size_t next_capacity = trace->capacity == 0 ? 1024 : trace->capacity * 2;
        ReplaySample *next = calloc(next_capacity, sizeof(*next));
        if (next == NULL) {
            return 0;
        }
        if (trace->count != 0) {
            memcpy(next, trace->items, trace->count * sizeof(*next));
        }
        free(trace->items);
        trace->items = next;
        trace->capacity = next_capacity;
    }

    trace->items[trace->count++] = sample;
    return 1;
}

static void free_replay_trace(ReplayTrace *trace)
{
    free(trace->items);
    memset(trace, 0, sizeof(*trace));
}

static int load_replay_trace(const char *path, ReplayTrace *trace)
{
    memset(trace, 0, sizeof(*trace));
    strcpy(trace->parent_hash, "unknown");
    strcpy(trace->rule_hash, "unknown");

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    char line[16384];
    uint32_t action_ids[ACTION_COUNT];
    nn_real scores[ACTION_COUNT];
    nn_real probabilities[ACTION_COUNT];
    int ok = 1;
    while (ok && fgets(line, sizeof(line), fp) != NULL) {
        NNParsedTraceRecord record;
        uint64_t p = 0;
        nn_real log_scale = (nn_real)log((double)DEFAULT_TEST_LIMIT + (double)G_MAX);

        if (!nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  ACTION_COUNT)) {
            ok = 0;
            break;
        }
        if (strcmp(record.task_name, "prime_gap") != 0) {
            continue;
        }
        if (record.candidate_count != ACTION_COUNT ||
            !nn_trace_json_get_u64(line, "p", &p)) {
            ok = 0;
            break;
        }
        nn_trace_json_get_real(line, "log_scale", &log_scale);
        if (strcmp(trace->parent_hash, "unknown") == 0 &&
            strcmp(record.model_hash_hex, "unknown") != 0) {
            strcpy(trace->parent_hash, record.model_hash_hex);
        }
        if (strcmp(trace->rule_hash, "unknown") == 0) {
            char parsed_rule_hash[65];
            if (nn_trace_json_get_string(line,
                                         "rule_set_hash",
                                         parsed_rule_hash,
                                         sizeof(parsed_rule_hash)) &&
                parsed_rule_hash[0] != '\0') {
                strcpy(trace->rule_hash, parsed_rule_hash);
            }
        }
        uint8_t seen_actions[ACTION_COUNT] = {0};
        for (size_t row = 0; row < record.candidate_count; ++row) {
            size_t action = ACTION_COUNT;
            if (!prime_action_from_action_id(action_ids[row], &action) ||
                seen_actions[action]) {
                ok = 0;
                break;
            }
            seen_actions[action] = 1;
        }
        if (!ok || p > UINT32_MAX || record.target_index >= ACTION_COUNT) {
            ok = 0;
            break;
        }

        ReplaySample sample;
        sample.p = (uint32_t)p;
        sample.target_action_id = action_ids[record.target_index];
        sample.log_scale = (double)log_scale;
        ok = append_replay_sample(trace, sample);
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || trace->count == 0) {
        free_replay_trace(trace);
        return 0;
    }
    return 1;
}

static void print_replay_usage(FILE *stream)
{
    fputs("usage: prime_replay trace.jsonl --out model.nnckpt [epochs] [--rules rules.crlprule]\n",
          stream);
}

int nn_prime_replay_main(int argc, char **argv)
{
    if (argc < 4) {
        print_replay_usage(stderr);
        return EXIT_FAILURE;
    }

    const char *trace_path = argv[1];
    const char *out_path = NULL;
    const char *rules_path = NULL;
    size_t epochs = 3;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--out") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_replay_usage(stderr);
                return EXIT_FAILURE;
            }
            out_path = argv[i];
        } else if (strcmp(argv[i], "--rules") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_replay_usage(stderr);
                return EXIT_FAILURE;
            }
            rules_path = argv[i];
        } else {
            char *end = NULL;
            unsigned long value = strtoul(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || value == 0) {
                print_replay_usage(stderr);
                return EXIT_FAILURE;
            }
            epochs = (size_t)value;
        }
    }
    if (out_path == NULL) {
        print_replay_usage(stderr);
        return EXIT_FAILURE;
    }

    SymbolicRuleSet replay_rules = {0};
    char rule_hash[65] = "unknown";
    int use_rules = 0;
    if (rules_path != NULL) {
        if (!load_symbolic_rules_with_hash(rules_path, &replay_rules, rule_hash)) {
            fprintf(stderr, "failed to load symbolic rules: %s\n", rules_path);
            print_replay_usage(stderr);
            return EXIT_FAILURE;
        }
        use_rules = 1;
        printf("loaded replay symbolic rules: %s hash=%s rules=%zu\n",
               rules_path,
               rule_hash,
               replay_rules.count);
    }

    ReplayTrace trace;
    if (!load_replay_trace(trace_path, &trace)) {
        fprintf(stderr, "failed to load replay trace: %s\n", trace_path);
        free_symbolic_rule_set(&replay_rules);
        return EXIT_FAILURE;
    }
    if (!use_rules && strcmp(trace.rule_hash, "unknown") != 0) {
        strcpy(rule_hash, trace.rule_hash);
    }

    const size_t layers[] = {FEATURE_COUNT, NN_PRIME_GAP_HIDDEN1, NN_PRIME_GAP_HIDDEN2, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 2026;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    nn_real *candidate_inputs = calloc((size_t)ACTION_COUNT * FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    if (net == NULL || candidate_inputs == NULL) {
        fputs("failed to allocate replay model\n", stderr);
        nn_free(net);
        free(candidate_inputs);
        free_replay_trace(&trace);
        free_symbolic_rule_set(&replay_rules);
        return EXIT_FAILURE;
    }

    size_t survivor_actions[ACTION_COUNT];
    uint32_t current_action_ids[ACTION_COUNT];
    fill_action_ids(current_action_ids);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < trace.count; ++i) {
            ReplaySample sample = trace.items[i];
            size_t candidate_count = ACTION_COUNT;
            size_t target_index = action_row_for_action_id(current_action_ids,
                                                           ACTION_COUNT,
                                                           sample.target_action_id);
            if (use_rules) {
                size_t target_action = ACTION_COUNT;
                candidate_count = fill_symbolic_survivor_candidates(&replay_rules,
                                                                    candidate_inputs,
                                                                    sample.p,
                                                                    sample.log_scale,
                                                                    &FEATURE_MASK_FULL,
                                                                    survivor_actions);
                if (!prime_action_from_action_id(sample.target_action_id, &target_action)) {
                    fputs("replay trace target action_id is unknown\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_symbolic_rule_set(&replay_rules);
                    return EXIT_FAILURE;
                }
                target_index = survivor_index_for_action(survivor_actions,
                                                         candidate_count,
                                                         target_action);
                if (candidate_count == 0 || target_index == candidate_count) {
                    fputs("replay symbolic rules rejected the target action\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_symbolic_rule_set(&replay_rules);
                    return EXIT_FAILURE;
                }
            } else {
                fill_candidate_features_masked(candidate_inputs,
                                               sample.p,
                                               sample.log_scale,
                                               &FEATURE_MASK_FULL);
                if (target_index == ACTION_COUNT) {
                    fputs("replay trace target action_id is not in current candidates\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_symbolic_rule_set(&replay_rules);
                    return EXIT_FAILURE;
                }
            }
            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       candidate_count,
                                                       target_index,
                                                       (nn_real)0.005);
            if (isnan((double)step_loss)) {
                fputs("replay training failed\n", stderr);
                nn_free(net);
                free(candidate_inputs);
                free_replay_trace(&trace);
                free_symbolic_rule_set(&replay_rules);
                return EXIT_FAILURE;
            }
            loss += (double)step_loss;
        }
        printf("replay epoch %zu loss=%.4f rows=%zu\n",
               epoch + 1,
               loss / (double)trace.count,
               trace.count);
    }

    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = 1;
    strcpy(meta.parent_model_hash, trace.parent_hash);
    snprintf(meta.training_trace_root,
             sizeof(meta.training_trace_root),
             "%s",
             strcmp(rule_hash, "unknown") == 0 ? "jsonl" : rule_hash);
    snprintf(meta.notes,
             sizeof(meta.notes),
             "%s",
             use_rules ? "prime_gap successor trained from learned-wheel trace replay"
                       : "prime_gap successor trained from trace replay");

    char hash[65];
    int ok = nn_save_checkpoint(net, out_path, &meta) &&
             nn_hash_checkpoint_hex(out_path, hash);
    if (ok) {
        printf("saved replay checkpoint: %s hash=%s parent=%s\n",
               out_path,
               hash,
               trace.parent_hash);
    }

    nn_free(net);
    free(candidate_inputs);
    free_replay_trace(&trace);
    free_symbolic_rule_set(&replay_rules);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

static void print_compare_usage(FILE *stream)
{
    fputs("usage: prime_compare [--rules rules.crlprule] "
          "model_a.nnckpt model_b.nnckpt [model_c.nnckpt ...]\n",
          stream);
}

static void print_eval_reliability_bins_json(const EvalMetrics *metrics)
{
    fputs(",\"reliability_bins\":[", stdout);
    int first = 1;
    for (size_t bin = 0; bin < CALIBRATION_BINS; ++bin) {
        if (metrics->calibration_count[bin] == 0) {
            continue;
        }
        printf("%s{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
               "\"confidence\":%.8f,\"accuracy\":%.8f}",
               first ? "" : ",",
               (double)bin / (double)CALIBRATION_BINS,
               (double)(bin + 1) / (double)CALIBRATION_BINS,
               metrics->calibration_count[bin],
               metrics->calibration_confidence[bin],
               metrics->calibration_correct[bin]);
        first = 0;
    }
    fputc(']', stdout);
}

static void print_report_json(const char *model_hash,
                              const NNCheckpointMeta *meta,
                              uint32_t seed,
                              const char *evaluation_mode,
                              const char *rule_hash,
                              EvalMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"prime_gap\","
           "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
           "\"evaluation_mode\":\"%s\",\"rule_set_hash\":\"%s\","
           "\"feature_encoder_version\":%u,"
           "\"benchmark_seed\":%u,\"sample_count\":%zu,"
           "\"top1\":%.8f,\"top5\":%.8f,\"top10\":%.8f,"
           "\"nll\":%.8f,\"mrr\":%.8f,\"ece\":%.8f,"
           "\"mean_rank\":%.8f,\"proposal_steps\":%.8f,"
           "\"wheel_exact\":%.8f,\"assisted_wheel\":%.8f,"
           "\"block_cost\":%.8f,\"assisted_block\":%.8f,"
           "\"model_ms_per_sample\":%.8f,\"verifier_ms_per_sample\":%.8f",
           model_hash,
           meta->parent_model_hash[0] == '\0' ? "unknown" : meta->parent_model_hash,
           evaluation_mode == NULL ? "raw" : evaluation_mode,
           rule_hash == NULL ? "unknown" : rule_hash,
           meta->feature_encoder_version,
           seed,
           metrics.samples,
           (double)metrics.top1 / samples,
           (double)metrics.top5 / samples,
           (double)metrics.top10 / samples,
           metrics.nll,
           metrics.mean_reciprocal_rank,
           metrics.expected_calibration_error,
           metrics.mean_true_rank,
           metrics.proposal_steps_top10,
           metrics.wheel_scan_exact_tests,
           metrics.assisted_wheel_exact_tests_top10,
           metrics.block_scan_cost,
           metrics.assisted_block_cost_top10,
           metrics.samples == 0 ? 0.0
                                : 1000.0 * metrics.scorer_seconds /
                                      (double)metrics.samples,
           metrics.verifier_samples == 0 ? 0.0
                                          : 1000.0 * metrics.verifier_seconds /
                                                (double)metrics.verifier_samples);
    print_eval_reliability_bins_json(&metrics);
    printf("}\n");
}

static void print_symbolic_compare_json(const char *rule_hash,
                                        uint32_t seed,
                                        SymbolicWheelMetrics metrics)
{
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"prime_gap\","
           "\"model_hash\":\"none\",\"parent_hash\":\"unknown\","
           "\"evaluation_mode\":\"learned_wheel_rules\","
           "\"rule_set_hash\":\"%s\",\"feature_encoder_version\":1,"
           "\"benchmark_seed\":%u,\"sample_count\":%zu,"
           "\"accepted_rules\":%zu,\"false_rejections\":%zu,"
           "\"direct_exact\":%.8f,\"hand_wheel_exact\":%.8f,"
           "\"learned_wheel_exact\":%.8f,\"learned_wheel_mod\":%.8f}\n",
           rule_hash == NULL ? "unknown" : rule_hash,
           seed,
           metrics.samples,
           metrics.accepted_rules,
           metrics.test_false_rejections,
           metrics.direct_exact_tests,
           metrics.hand_wheel_exact_tests,
           metrics.learned_exact_tests,
           metrics.learned_mod_tests);
}

int nn_prime_compare_main(int argc, char **argv)
{
    const char *rules_path = NULL;
    size_t model_count = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rules") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_compare_usage(stderr);
                return EXIT_FAILURE;
            }
            rules_path = argv[i];
        } else if (argv[i][0] == '-') {
            print_compare_usage(stderr);
            return EXIT_FAILURE;
        } else {
            ++model_count;
        }
    }

    if (model_count < 2) {
        print_compare_usage(stderr);
        return EXIT_FAILURE;
    }

    const uint32_t train_limit = DEFAULT_TRAIN_LIMIT;
    const uint32_t test_limit = DEFAULT_TEST_LIMIT;
    const uint32_t sieve_limit = test_limit + (uint32_t)G_MAX;
    const uint32_t benchmark_seed = 20240511u;
    const double log_scale = log((double)test_limit + (double)G_MAX);

    SymbolicRuleSet compare_rules = {0};
    char rule_hash[65] = "unknown";
    int use_rules = 0;
    if (rules_path != NULL) {
        if (!load_symbolic_rules_with_hash(rules_path, &compare_rules, rule_hash)) {
            fprintf(stderr, "failed to load symbolic rules: %s\n", rules_path);
            return EXIT_FAILURE;
        }
        use_rules = 1;
    }

    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    if (!build_sieve(sieve_limit, &sieve) ||
        !build_gap_datasets(&sieve, train_limit, test_limit, &train, &validation, &test)) {
        fputs("failed to build benchmark data\n", stderr);
        free_symbolic_rule_set(&compare_rules);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    nn_real *candidate_inputs = calloc((size_t)ACTION_COUNT * FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(ACTION_COUNT, sizeof(*probabilities));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL) {
        fputs("failed to allocate benchmark buffers\n", stderr);
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        free_symbolic_rule_set(&compare_rules);
        free_gap_dataset(&train);
        free_gap_dataset(&validation);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    puts("jsonl report:");
    if (use_rules) {
        SymbolicWheelMetrics symbolic_metrics;
        memset(&symbolic_metrics, 0, sizeof(symbolic_metrics));
        symbolic_metrics.accepted_rules = compare_rules.count;
        symbolic_metrics = evaluate_symbolic_wheel(&sieve,
                                                   &compare_rules,
                                                   &test,
                                                   DEFAULT_MAX_TEST_STATES,
                                                   symbolic_metrics);
        print_symbolic_compare_json(rule_hash, benchmark_seed, symbolic_metrics);
        printf("learned wheel rules hash=%s direct_exact=%.2f hand_wheel_exact=%.2f "
               "learned_wheel_exact=%.2f false_rejections=%zu\n",
               rule_hash,
               symbolic_metrics.direct_exact_tests,
               symbolic_metrics.hand_wheel_exact_tests,
               symbolic_metrics.learned_exact_tests,
               symbolic_metrics.test_false_rejections);
    }
    int ok = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rules") == 0) {
            ++i;
            continue;
        }
        NNCheckpointMeta meta;
        NeuralNet *net = nn_load_checkpoint(argv[i], &meta);
        char hash[65] = "unknown";
        if (net == NULL) {
            fprintf(stderr, "failed to load checkpoint: %s\n", argv[i]);
            ok = 0;
            continue;
        }
        nn_hash_checkpoint_hex(argv[i], hash);
        EvalMetrics metrics = evaluate_model(net,
                                             &sieve,
                                             &test,
                                             DEFAULT_MAX_TEST_STATES,
                                             log_scale,
                                             &FEATURE_MASK_FULL,
                                             1,
                                             1.0,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        print_report_json(hash, &meta, benchmark_seed, "raw", "unknown", metrics);
        print_comparison_row(argv[i], 1.0, metrics);
        if (use_rules) {
            EvalMetrics learned_metrics =
                evaluate_model_symbolic_masked(net,
                                               &sieve,
                                               &compare_rules,
                                               &test,
                                               DEFAULT_MAX_TEST_STATES,
                                               log_scale,
                                               &FEATURE_MASK_FULL,
                                               1,
                                               1.0,
                                               candidate_inputs,
                                               scores,
                                               probabilities);
            print_report_json(hash,
                              &meta,
                              benchmark_seed,
                              "learned_wheel_model",
                              rule_hash,
                              learned_metrics);
            print_comparison_row("learned-wheel+model", 1.0, learned_metrics);
        }
        nn_free(net);
    }

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free_symbolic_rule_set(&compare_rules);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
