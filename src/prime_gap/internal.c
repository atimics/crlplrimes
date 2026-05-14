/* Shared constants, config parsing, and utility helpers. */

#include "internal.h"

const double DEFAULT_REGULARIZED_RL_REFERENCE_WEIGHT = 1.20;
const double DEFAULT_REGULARIZED_RL_ENTROPY_WEIGHT = 0.12;
const uint32_t FEATURE_MODS[] = {
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

const FeatureMask FEATURE_MASK_FULL = {"full", 1, 1, 1, 1};
const FeatureMask FEATURE_MASK_LOG_ONLY = {"log-only", 1, 0, 0, 0};
const FeatureMask FEATURE_MASK_GAP_ONLY = {"gap-only", 0, 1, 1, 0};
const FeatureMask FEATURE_MASK_RESIDUE_ONLY = {"residue-only", 0, 0, 1, 1};

double now_seconds(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

uint32_t rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

double rng_uniform(uint32_t *state)
{
    return (double)rng_next(state) / (double)UINT32_MAX;
}

size_t rng_bounded(uint32_t *state, size_t bound)
{
    return bound == 0 ? 0 : (size_t)(rng_next(state) % (uint32_t)bound);
}

int parse_size_arg(int argc, char **argv, int index, size_t fallback, size_t *value_out)
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

int parse_double_arg(int argc, char **argv, int index, double fallback, double *value_out)
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

int parse_size_text(const char *text, size_t *value_out)
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

int parse_u32_text(const char *text, uint32_t *value_out)
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

int parse_run_config(int argc, char **argv, PrimeGapRunConfig *config)
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

int run_config_is_valid(const PrimeGapRunConfig *config)
{
    return config->test_limit_arg > config->train_limit_arg &&
           config->train_limit_arg >= 16 &&
           config->test_limit_arg <= (size_t)UINT32_MAX - (size_t)G_MAX &&
           config->train_limit_arg <= (size_t)UINT32_MAX &&
           config->rng_seed_arg <= (size_t)UINT32_MAX &&
           config->regularized_reference_weight >= 0.0 &&
           config->regularized_entropy_weight >= 0.0;
}

void print_usage(FILE *stream)
{
    fputs("usage: prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] "
          "[max_train_states] [max_test_states] [regularized_reference_weight] "
          "[regularized_entropy_weight] [rng_seed] "
          "[--trace path] [--save path] [--model checkpoint] "
          "[--save-rules path] [--rules path] [--save-learned-model path]\n",
          stream);
}
