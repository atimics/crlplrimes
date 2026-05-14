/* Shared includes, constants, task types, config parsing, and utility helpers.
 * Included by ../nn_tasks_prime_gap.c; not compiled as a standalone translation unit.
 */

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

enum {
    RULE_SEARCH_GRAMMAR_VERSION = 2,
    RULE_SEARCH_MAX_MODULI = 32,
    RULE_SEARCH_FEATURE_COUNT = 9,
    RULE_SEARCH_SCORER_EPOCHS = 24,
};

typedef enum {
    RULE_FILTER_CANDIDATE_MOD = 0,
    RULE_FILTER_GAP_MOD = 1,
} RuleFilterKind;

typedef struct {
    uint32_t modulus;
    uint32_t residue;
} RuleStateAtom;

typedef struct {
    RuleFilterKind kind;
    uint32_t modulus;
    uint32_t residue;
} RuleFilterAtom;

typedef struct {
    RuleFilterKind kind;
    uint32_t state_modulus;
    uint32_t state_residue;
    uint32_t filter_modulus;
    uint32_t filter_residue;
    size_t train_rejects;
    size_t train_false_rejections;
    size_t validation_rejects;
    size_t validation_false_rejections;
    nn_real model_score;
    double utility;
    int accepted;
} ExpandedRule;

typedef struct {
    ExpandedRule *items;
    size_t count;
} ExpandedRuleSet;

typedef struct {
    RuleStateAtom *state_atoms;
    RuleFilterAtom *filter_atoms;
    size_t *state_offsets;
    size_t *candidate_offsets;
    size_t *gap_offsets;
    size_t state_count;
    size_t filter_count;
} RuleSearchGrammar;

typedef struct {
    size_t train_limit;
    size_t test_limit;
    size_t max_train_states;
    size_t max_test_states;
    size_t max_rules;
    uint32_t moduli[RULE_SEARCH_MAX_MODULI];
    size_t mod_count;
    int include_candidate_filters;
    int include_gap_filters;
    const char *trace_path;
    const char *save_path;
} RuleSearchConfig;

typedef struct {
    size_t proposed_rules;
    size_t verified_rules;
    size_t accepted_rules;
    size_t accepted_conditional_rules;
    size_t accepted_gap_rules;
    size_t accepted_candidate_rules;
    size_t unsafe_rules;
    size_t no_support_rules;
    size_t uncertified_rules;
    size_t samples;
    size_t false_rejections;
    double proposer_loss;
    double direct_exact_tests;
    double hand_wheel_exact_tests;
    double expanded_exact_tests;
    double expanded_mod_tests;
} RuleSearchMetrics;

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

static int parse_size_text(const char *text, size_t *value_out)
{
    if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
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

static int parse_u32_text(const char *text, uint32_t *value_out)
{
    size_t value = 0;
    if (!parse_size_text(text, &value) || value > (size_t)UINT32_MAX) {
        return 0;
    }
    *value_out = (uint32_t)value;
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
