/* Expanded rule grammar, verifier-gated rule selection, trace writing, and CLI entry point. */

#include "internal.h"

#include <sys/stat.h>
#include <unistd.h>

static const NNFeatureSchema PRIME_CLOSED_LOOP_RULE_FEATURES[RULE_DISCOVERY_FEATURE_COUNT] = {
    {"is_conditional", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"state_modulus", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"state_residue", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"state_residue_is_zero", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"state_common_factor", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_gap_filter", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"filter_modulus", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"filter_residue", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"filter_residue_is_zero", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"filter_common_factor", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"same_state_filter_modulus", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"candidate_residue", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"candidate_common_factor", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"inverse_filter_modulus", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"inverse_state_modulus", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"accepted_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"residual_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"accepted_candidate_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"accepted_gap_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"accepted_conditional_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_kind_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_filter_modulus_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_state_modulus_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_candidate_residue_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"is_subset_covered", NN_FEATURE_DERIVED_FROM_GATE, 1},
};

typedef struct {
    char magic[16];
    uint64_t key;
    uint64_t rule_count;
    uint64_t event_count;
    uint64_t word_count;
    uint64_t samples;
} RuleCoverageCacheHeader;

typedef struct {
    size_t hits;
    size_t misses;
    size_t writes;
} RuleCoverageCacheStats;

static RuleCoverageCacheStats RULE_COVERAGE_CACHE_STATS;

enum {
    UNIFIED_SCORER_FEATURE_COUNT = NN_SCORER_ENVELOPE_FEATURE_COUNT,
    UNIFIED_F_BIAS = NN_SCORER_ENVELOPE_F_BIAS,
    UNIFIED_F_ROLE_RULE = NN_SCORER_ENVELOPE_F_ROLE_RULE,
    UNIFIED_F_DOMAIN_PRIME_RULE = NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_RULE,
    UNIFIED_F_ROLE_ACTION = NN_SCORER_ENVELOPE_F_ROLE_ACTION,
    UNIFIED_F_DOMAIN_PRIME_ACTION = NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_ACTION,
    UNIFIED_F_RULE_INDEX = NN_SCORER_ENVELOPE_F_CANDIDATE_INDEX,
    UNIFIED_F_CANDIDATE_COUNT = NN_SCORER_ENVELOPE_F_CANDIDATE_COUNT,
    UNIFIED_F_DOMAIN_GRID_ACTION = NN_SCORER_ENVELOPE_F_DOMAIN_GRID_ACTION,
    UNIFIED_F_DOMAIN_CLAIM_ACTION = NN_SCORER_ENVELOPE_F_DOMAIN_CLAIM_ACTION,
    UNIFIED_F_DOMAIN_ARC_ACTION = NN_SCORER_ENVELOPE_F_DOMAIN_ARC_ACTION,
    UNIFIED_F_ACTION_COMMON = NN_SCORER_ENVELOPE_F_ACTION_COMMON,
    UNIFIED_F_ACTION_STATE = NN_SCORER_ENVELOPE_F_ACTION_STATE,
    UNIFIED_F_ACTION_PAYLOAD = NN_SCORER_ENVELOPE_F_ROLE_PAYLOAD,
    UNIFIED_F_RULE_FEATURES = NN_SCORER_ENVELOPE_F_RESIDUAL_FEATURES,
    UNIFIED_PRIME_RESIDUE_FEATURES = FEATURE_MOD_COUNT,
    UNIFIED_TRANSFER_MIN_RULE_UPDATES = 128,
    UNIFIED_TRANSFER_RULE_UPDATES_PER_BUDGET = 8,
};

PRIME_GAP_TEST_API RuleSearchConfig default_rule_search_config(void)
{
    RuleSearchConfig config;
    memset(&config, 0, sizeof(config));
    config.train_limit = DEFAULT_TRAIN_LIMIT;
    config.test_limit = DEFAULT_TEST_LIMIT;
    config.max_train_states = DEFAULT_MAX_TRAIN_STATES;
    config.max_test_states = DEFAULT_MAX_TEST_STATES;
    config.max_marginal_states = (size_t)RULE_SEARCH_MARGINAL_MAX_STATES;
    config.max_rules = 256;
    config.rule_hidden1 = 0;
    config.rule_hidden2 = 0;
    config.unified_hidden1 = 0;
    config.unified_hidden2 = 0;
    config.proposer_mode = RULE_PROPOSER_NEURAL;
    config.proposer_seed = 20240511u;
    config.include_state_conditions = 1;
    config.include_candidate_filters = 1;
    config.include_gap_filters = 1;
    config.mod_count = FEATURE_MOD_COUNT < RULE_SEARCH_MAX_MODULI
                           ? FEATURE_MOD_COUNT
                           : RULE_SEARCH_MAX_MODULI;
    for (size_t i = 0; i < config.mod_count; ++i) {
        config.moduli[i] = FEATURE_MODS[i];
    }
    return config;
}

PRIME_GAP_TEST_API size_t rule_search_effective_hidden1(const RuleSearchConfig *config)
{
    return config != NULL && config->rule_hidden1 != 0 ? config->rule_hidden1 : 32u;
}

PRIME_GAP_TEST_API size_t rule_search_effective_hidden2(const RuleSearchConfig *config)
{
    return config != NULL && config->rule_hidden2 != 0 ? config->rule_hidden2 : 16u;
}

static size_t rule_search_effective_unified_hidden1(const RuleSearchConfig *config)
{
    return config != NULL && config->unified_hidden1 != 0 ? config->unified_hidden1 : 64u;
}

static size_t rule_search_effective_unified_hidden2(const RuleSearchConfig *config)
{
    return config != NULL && config->unified_hidden2 != 0 ? config->unified_hidden2 : 32u;
}

PRIME_GAP_TEST_API void rule_search_closed_loop_hidden_json(const RuleSearchConfig *config,
                                                            char *buffer,
                                                            size_t buffer_size)
{
    if (buffer == NULL || buffer_size == 0) {
        return;
    }
    if (config != NULL && config->proposer_mode == RULE_PROPOSER_CLOSED_LOOP) {
        snprintf(buffer,
                 buffer_size,
                 "\"%zux%zu\"",
                 rule_search_effective_hidden1(config),
                 rule_search_effective_hidden2(config));
    } else {
        snprintf(buffer, buffer_size, "null");
    }
}

PRIME_GAP_TEST_API void rule_search_unified_hidden_json(const RuleSearchConfig *config,
                                                        char *buffer,
                                                        size_t buffer_size)
{
    if (buffer == NULL || buffer_size == 0) {
        return;
    }
    if (config != NULL && config->proposer_mode == RULE_PROPOSER_UNIFIED) {
        snprintf(buffer,
                 buffer_size,
                 "\"%zux%zu\"",
                 rule_search_effective_unified_hidden1(config),
                 rule_search_effective_unified_hidden2(config));
    } else {
        snprintf(buffer, buffer_size, "null");
    }
}

static const char *rule_proposer_mode_name(RuleProposerMode mode)
{
    switch (mode) {
    case RULE_PROPOSER_NEURAL:
        return "neural_structural";
    case RULE_PROPOSER_RANDOM:
        return "random";
    case RULE_PROPOSER_GRAMMAR:
        return "grammar_order";
    case RULE_PROPOSER_STRUCTURAL:
        return "structural_heuristic";
    case RULE_PROPOSER_VALIDATION_ORACLE:
        return "validation_oracle";
    case RULE_PROPOSER_CLOSED_LOOP:
        return "closed_loop_neural";
    case RULE_PROPOSER_UNIFIED:
        return "unified_scorer";
    }
    return "unknown";
}

PRIME_GAP_TEST_API const char *rule_proposer_feature_set_name(RuleProposerMode mode)
{
    switch (mode) {
    case RULE_PROPOSER_NEURAL:
        return "structural";
    case RULE_PROPOSER_RANDOM:
    case RULE_PROPOSER_GRAMMAR:
        return "none";
    case RULE_PROPOSER_STRUCTURAL:
        return "structural_heuristic";
    case RULE_PROPOSER_VALIDATION_ORACLE:
        return "validation_counts";
    case RULE_PROPOSER_CLOSED_LOOP:
        return "structural_residual_set";
    case RULE_PROPOSER_UNIFIED:
        return nn_scorer_envelope_feature_set();
    }
    return "unknown";
}

PRIME_GAP_TEST_API const char *rule_proposer_target_name(RuleProposerMode mode)
{
    switch (mode) {
    case RULE_PROPOSER_NEURAL:
        return "train_marginal_utility";
    case RULE_PROPOSER_CLOSED_LOOP:
        return "residual_marginal_utility";
    case RULE_PROPOSER_UNIFIED:
        return "joint_action_rule_softmax";
    case RULE_PROPOSER_VALIDATION_ORACLE:
        return "validation_marginal_utility";
    case RULE_PROPOSER_RANDOM:
    case RULE_PROPOSER_GRAMMAR:
    case RULE_PROPOSER_STRUCTURAL:
        return "none";
    }
    return "unknown";
}

static int parse_rule_proposer_mode(const char *text, RuleProposerMode *mode)
{
    if (strcmp(text, "neural") == 0 || strcmp(text, "neural-structural") == 0) {
        *mode = RULE_PROPOSER_NEURAL;
        return 1;
    }
    if (strcmp(text, "random") == 0) {
        *mode = RULE_PROPOSER_RANDOM;
        return 1;
    }
    if (strcmp(text, "grammar") == 0 || strcmp(text, "grammar-order") == 0 ||
        strcmp(text, "lexical") == 0) {
        *mode = RULE_PROPOSER_GRAMMAR;
        return 1;
    }
    if (strcmp(text, "structural") == 0 || strcmp(text, "structural-heuristic") == 0) {
        *mode = RULE_PROPOSER_STRUCTURAL;
        return 1;
    }
    if (strcmp(text, "validation-oracle") == 0 || strcmp(text, "oracle") == 0) {
        *mode = RULE_PROPOSER_VALIDATION_ORACLE;
        return 1;
    }
    if (strcmp(text, "closed-loop") == 0 || strcmp(text, "residual") == 0 ||
        strcmp(text, "neural-closed-loop") == 0) {
        *mode = RULE_PROPOSER_CLOSED_LOOP;
        return 1;
    }
    if (strcmp(text, "unified") == 0 || strcmp(text, "unified-scorer") == 0 ||
        strcmp(text, "unified_scorer") == 0) {
        *mode = RULE_PROPOSER_UNIFIED;
        return 1;
    }
    return 0;
}

static int parse_rule_search_moduli(const char *text,
                                    uint32_t *moduli,
                                    size_t *mod_count)
{
    if (text == NULL || text[0] == '\0') {
        return 0;
    }

    char *copy = malloc(strlen(text) + 1u);
    if (copy == NULL) {
        return 0;
    }
    strcpy(copy, text);

    size_t count = 0;
    int ok = 1;
    for (char *part = strtok(copy, ","); part != NULL; part = strtok(NULL, ",")) {
        uint32_t modulus = 0;
        if (!parse_u32_text(part, &modulus) || modulus < 3 ||
            modulus > (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD ||
            count >= RULE_SEARCH_MAX_MODULI) {
            ok = 0;
            break;
        }
        for (size_t i = 0; i < count; ++i) {
            if (moduli[i] == modulus) {
                ok = 0;
                break;
            }
        }
        if (!ok) {
            break;
        }
        moduli[count++] = modulus;
    }
    free(copy);
    if (!ok || count == 0) {
        return 0;
    }

    for (size_t i = 1; i < count; ++i) {
        uint32_t value = moduli[i];
        size_t j = i;
        while (j > 0 && moduli[j - 1] > value) {
            moduli[j] = moduli[j - 1];
            --j;
        }
        moduli[j] = value;
    }

    *mod_count = count;
    return 1;
}

static int parse_rule_search_moduli_range(const char *text,
                                          uint32_t *moduli,
                                          size_t *mod_count)
{
    if (text == NULL || text[0] == '\0') {
        return 0;
    }

    char copy[64];
    if (strlen(text) >= sizeof(copy)) {
        return 0;
    }
    strcpy(copy, text);

    char *separator = strstr(copy, "..");
    if (separator != NULL) {
        *separator = '\0';
        separator += 2;
    } else {
        separator = strchr(copy, ':');
        if (separator == NULL) {
            return 0;
        }
        *separator++ = '\0';
    }

    uint32_t first = 0;
    uint32_t last = 0;
    if (!parse_u32_text(copy, &first) || !parse_u32_text(separator, &last) ||
        first < 3 || first > last || last > (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD) {
        return 0;
    }

    size_t count = (size_t)(last - first + 1u);
    if (count == 0 || count > RULE_SEARCH_MAX_MODULI) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        moduli[i] = first + (uint32_t)i;
    }
    *mod_count = count;
    return 1;
}

static int parse_rule_hidden_text(const char *text, size_t *hidden1, size_t *hidden2)
{
    if (text == NULL || hidden1 == NULL || hidden2 == NULL || text[0] == '\0') {
        return 0;
    }

    char copy[64];
    size_t length = strlen(text);
    if (length >= sizeof(copy)) {
        return 0;
    }
    memcpy(copy, text, length + 1u);

    char *separator = strchr(copy, 'x');
    if (separator == NULL) {
        separator = strchr(copy, 'X');
    }
    if (separator == NULL) {
        separator = strchr(copy, '/');
    }
    if (separator == NULL) {
        separator = strchr(copy, ',');
    }
    if (separator == NULL) {
        return 0;
    }
    *separator++ = '\0';

    size_t first = 0;
    size_t second = 0;
    if (!parse_size_text(copy, &first) || !parse_size_text(separator, &second) ||
        first == 0 || second == 0) {
        return 0;
    }
    *hidden1 = first;
    *hidden2 = second;
    return 1;
}

PRIME_GAP_TEST_API int parse_rule_search_config(int argc, char **argv, RuleSearchConfig *config)
{
    *config = default_rule_search_config();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--train-limit") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->train_limit)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--test-limit") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->test_limit)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-train-states") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->max_train_states)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-test-states") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->max_test_states)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-marginal-states") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->max_marginal_states)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-rules") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->max_rules)) {
                return 0;
            }
        } else if ((strcmp(argv[i], "--rule-hidden") == 0 ||
                    strcmp(argv[i], "--closed-loop-hidden") == 0) &&
                   i + 1 < argc) {
            if (!parse_rule_hidden_text(argv[++i],
                                        &config->rule_hidden1,
                                        &config->rule_hidden2)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--unified-scorer-hidden") == 0 && i + 1 < argc) {
            if (!parse_rule_hidden_text(argv[++i],
                                        &config->unified_hidden1,
                                        &config->unified_hidden2)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--proposer") == 0 && i + 1 < argc) {
            if (!parse_rule_proposer_mode(argv[++i], &config->proposer_mode)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--proposer-seed") == 0 && i + 1 < argc) {
            size_t seed = 0;
            if (!parse_size_text(argv[++i], &seed)) {
                return 0;
            }
            config->proposer_seed = (uint64_t)seed;
        } else if (strcmp(argv[i], "--moduli") == 0 && i + 1 < argc) {
            if (!parse_rule_search_moduli(argv[++i], config->moduli, &config->mod_count)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--moduli-range") == 0 && i + 1 < argc) {
            if (!parse_rule_search_moduli_range(argv[++i],
                                                config->moduli,
                                                &config->mod_count)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--unconditional-only") == 0) {
            config->include_state_conditions = 0;
        } else if (strcmp(argv[i], "--gap-only") == 0) {
            config->include_candidate_filters = 0;
            config->include_gap_filters = 1;
        } else if (strcmp(argv[i], "--candidate-only") == 0) {
            config->include_candidate_filters = 1;
            config->include_gap_filters = 0;
        } else if (strcmp(argv[i], "--coverage-cache-dir") == 0 && i + 1 < argc) {
            config->coverage_cache_dir = argv[++i];
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else {
            return 0;
        }
    }

    return config->test_limit > config->train_limit &&
           config->train_limit >= 16 &&
           config->test_limit <= (size_t)UINT32_MAX - (size_t)G_MAX &&
           config->max_marginal_states > 0 &&
           config->max_rules > 0 &&
           config->mod_count > 0 &&
           (config->include_candidate_filters || config->include_gap_filters);
}

static void print_rule_search_usage(FILE *stream)
{
    fputs("usage: prime_rule_search [--train-limit n] [--test-limit n] "
          "[--max-train-states n] [--max-test-states n] "
          "[--max-marginal-states n] [--max-rules n] "
          "[--closed-loop-hidden h1xh2] [--unified-scorer-hidden h1xh2] "
          "[--proposer neural|closed-loop|unified|random|grammar|structural|validation-oracle] "
          "[--proposer-seed n] "
          "[--gap-only|--candidate-only] [--unconditional-only] "
          "[--moduli 3,5,7,...] [--moduli-range 3:255] "
          "[--coverage-cache-dir dir] [--trace path] [--save path]\n",
          stream);
}

static const char *rule_filter_kind_name(RuleFilterKind kind)
{
    return kind == RULE_FILTER_GAP_MOD ? "gap_mod" : "candidate_mod";
}

static int expanded_rule_is_conditional(const ExpandedRule *rule)
{
    return rule->state_modulus != 1u;
}

static int expanded_rule_is_derived_divisibility(const ExpandedRule *rule)
{
    return rule->kind == RULE_FILTER_GAP_MOD &&
           rule->state_modulus == rule->filter_modulus &&
           ((rule->state_residue + rule->filter_residue) % rule->filter_modulus) == 0u;
}

static int expanded_rule_has_deterministic_certificate(const ExpandedRule *rule)
{
    if (rule->kind == RULE_FILTER_CANDIDATE_MOD) {
        return gcd_u32(rule->filter_modulus, rule->filter_residue) > 1u;
    }
    if (rule->state_modulus != rule->filter_modulus) {
        return 0;
    }
    uint32_t candidate_residue =
        (rule->state_residue + rule->filter_residue) % rule->filter_modulus;
    return gcd_u32(rule->filter_modulus, candidate_residue) > 1u;
}

static uint32_t expanded_rule_structural_candidate_residue(const ExpandedRule *rule)
{
    if (rule->kind == RULE_FILTER_CANDIDATE_MOD) {
        return rule->filter_residue;
    }
    if (rule->state_modulus == rule->filter_modulus) {
        return (rule->state_residue + rule->filter_residue) % rule->filter_modulus;
    }
    return rule->filter_residue;
}

static int format_expanded_rule_text(char *buffer, size_t buffer_size, const ExpandedRule *rule)
{
    if (buffer == NULL || buffer_size == 0 || rule == NULL) {
        return 0;
    }
    int written = 0;
    if (expanded_rule_is_conditional(rule)) {
        written = snprintf(buffer,
                           buffer_size,
                           "if p %% %u == %u then reject %s %% %u == %u",
                           rule->state_modulus,
                           rule->state_residue,
                           rule->kind == RULE_FILTER_GAP_MOD ? "gap" : "candidate",
                           rule->filter_modulus,
                           rule->filter_residue);
    } else {
        written = snprintf(buffer,
                           buffer_size,
                           "reject %s %% %u == %u",
                           rule->kind == RULE_FILTER_GAP_MOD ? "gap" : "candidate",
                           rule->filter_modulus,
                           rule->filter_residue);
    }
    return written >= 0 && (size_t)written < buffer_size;
}

static void print_expanded_rule_text(FILE *fp, const ExpandedRule *rule)
{
    char text[160];
    if (format_expanded_rule_text(text, sizeof(text), rule)) {
        fputs(text, fp);
    }
}

PRIME_GAP_TEST_API void free_rule_search_grammar(RuleSearchGrammar *grammar)
{
    free(grammar->state_atoms);
    free(grammar->filter_atoms);
    free(grammar->state_offsets);
    free(grammar->candidate_offsets);
    free(grammar->gap_offsets);
    memset(grammar, 0, sizeof(*grammar));
}

PRIME_GAP_TEST_API int build_rule_search_grammar(const RuleSearchConfig *config,
                                                 RuleSearchGrammar *grammar)
{
    memset(grammar, 0, sizeof(*grammar));
    size_t residue_sum = 0;
    for (size_t i = 0; i < config->mod_count; ++i) {
        residue_sum += config->moduli[i];
    }

    size_t filter_kinds = (config->include_candidate_filters ? 1u : 0u) +
                          (config->include_gap_filters ? 1u : 0u);
    grammar->state_count = (config->include_state_conditions ? residue_sum : 0u) + 1u;
    grammar->filter_count = residue_sum * filter_kinds;
    grammar->state_atoms = calloc(grammar->state_count, sizeof(*grammar->state_atoms));
    grammar->filter_atoms = calloc(grammar->filter_count, sizeof(*grammar->filter_atoms));
    grammar->state_offsets = calloc(config->mod_count, sizeof(*grammar->state_offsets));
    grammar->candidate_offsets = calloc(config->mod_count, sizeof(*grammar->candidate_offsets));
    grammar->gap_offsets = calloc(config->mod_count, sizeof(*grammar->gap_offsets));
    if (grammar->state_atoms == NULL || grammar->filter_atoms == NULL ||
        grammar->state_offsets == NULL || grammar->candidate_offsets == NULL ||
        grammar->gap_offsets == NULL) {
        free_rule_search_grammar(grammar);
        return 0;
    }
    for (size_t i = 0; i < config->mod_count; ++i) {
        grammar->candidate_offsets[i] = (size_t)-1;
        grammar->gap_offsets[i] = (size_t)-1;
    }

    grammar->state_atoms[0].modulus = 1;
    grammar->state_atoms[0].residue = 0;
    size_t state_index = 1;
    size_t filter_index = 0;
    for (size_t mod_index = 0; mod_index < config->mod_count; ++mod_index) {
        uint32_t modulus = config->moduli[mod_index];
        grammar->state_offsets[mod_index] = (size_t)-1;
        if (config->include_state_conditions) {
            grammar->state_offsets[mod_index] = state_index;
            for (uint32_t residue = 0; residue < modulus; ++residue) {
                grammar->state_atoms[state_index].modulus = modulus;
                grammar->state_atoms[state_index].residue = residue;
                ++state_index;
            }
        }

        if (config->include_candidate_filters) {
            grammar->candidate_offsets[mod_index] = filter_index;
            for (uint32_t residue = 0; residue < modulus; ++residue) {
                grammar->filter_atoms[filter_index].kind = RULE_FILTER_CANDIDATE_MOD;
                grammar->filter_atoms[filter_index].modulus = modulus;
                grammar->filter_atoms[filter_index].residue = residue;
                ++filter_index;
            }
        }

        if (config->include_gap_filters) {
            grammar->gap_offsets[mod_index] = filter_index;
            for (uint32_t residue = 0; residue < modulus; ++residue) {
                grammar->filter_atoms[filter_index].kind = RULE_FILTER_GAP_MOD;
                grammar->filter_atoms[filter_index].modulus = modulus;
                grammar->filter_atoms[filter_index].residue = residue;
                ++filter_index;
            }
        }
    }

    return state_index == grammar->state_count && filter_index == grammar->filter_count;
}

PRIME_GAP_TEST_API int build_expanded_rules(const RuleSearchGrammar *grammar,
                                            ExpandedRuleSet *rules)
{
    memset(rules, 0, sizeof(*rules));
    rules->count = grammar->state_count * grammar->filter_count;
    rules->items = calloc(rules->count, sizeof(*rules->items));
    if (rules->items == NULL) {
        return 0;
    }

    size_t index = 0;
    for (size_t state = 0; state < grammar->state_count; ++state) {
        for (size_t filter = 0; filter < grammar->filter_count; ++filter) {
            ExpandedRule *rule = &rules->items[index++];
            rule->state_modulus = grammar->state_atoms[state].modulus;
            rule->state_residue = grammar->state_atoms[state].residue;
            rule->kind = grammar->filter_atoms[filter].kind;
            rule->filter_modulus = grammar->filter_atoms[filter].modulus;
            rule->filter_residue = grammar->filter_atoms[filter].residue;
        }
    }
    return 1;
}

void free_expanded_rule_set(ExpandedRuleSet *rules)
{
    free(rules->items);
    memset(rules, 0, sizeof(*rules));
}

static void increment_matching_expanded_rules(ExpandedRule *rules,
                                             const RuleSearchConfig *config,
                                             const RuleSearchGrammar *grammar,
                                             uint32_t p,
                                             uint32_t gap,
                                             int false_rejection,
                                             int validation)
{
    size_t state_indices[RULE_SEARCH_MAX_MODULI + 1u];
    size_t filter_indices[RULE_SEARCH_MAX_MODULI * 2u];
    size_t state_count = 1;
    size_t filter_count = 0;
    uint32_t candidate = p + gap;

    state_indices[0] = 0;
    for (size_t i = 0; i < config->mod_count; ++i) {
        uint32_t modulus = config->moduli[i];
        if (grammar->state_offsets[i] != (size_t)-1) {
            state_indices[state_count++] = grammar->state_offsets[i] + (p % modulus);
        }
        if (candidate > modulus) {
            if (grammar->candidate_offsets[i] != (size_t)-1) {
                filter_indices[filter_count++] =
                    grammar->candidate_offsets[i] + (candidate % modulus);
            }
            if (grammar->gap_offsets[i] != (size_t)-1) {
                filter_indices[filter_count++] = grammar->gap_offsets[i] + (gap % modulus);
            }
        }
    }

    for (size_t state = 0; state < state_count; ++state) {
        size_t base = state_indices[state] * grammar->filter_count;
        for (size_t filter = 0; filter < filter_count; ++filter) {
            ExpandedRule *rule = &rules[base + filter_indices[filter]];
            if (false_rejection) {
                if (validation) {
                    ++rule->validation_false_rejections;
                } else {
                    ++rule->train_false_rejections;
                }
            } else if (validation) {
                ++rule->validation_rejects;
            } else {
                ++rule->train_rejects;
            }
        }
    }
}

PRIME_GAP_TEST_API void accumulate_expanded_rule_stats(ExpandedRuleSet *rules,
                                                       const RuleSearchConfig *config,
                                                       const RuleSearchGrammar *grammar,
                                                       const GapDataset *dataset,
                                                       size_t max_states,
                                                       int validation)
{
    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t true_gap = dataset->gap[index];
        for (uint32_t gap = 2; gap < true_gap; gap += 2) {
            increment_matching_expanded_rules(rules->items,
                                             config,
                                             grammar,
                                             p,
                                             gap,
                                             0,
                                             validation);
        }
        increment_matching_expanded_rules(rules->items,
                                         config,
                                         grammar,
                                         p,
                                         true_gap,
                                         1,
                                         validation);
    }
}

static size_t rule_search_sample_count(const GapDataset *dataset, size_t max_states)
{
    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }
    return samples;
}

PRIME_GAP_TEST_API size_t rule_search_marginal_state_limit(const RuleSearchConfig *config)
{
    if (config->max_train_states != 0 &&
        config->max_train_states < config->max_marginal_states) {
        return config->max_train_states;
    }
    return config->max_marginal_states;
}

static size_t count_negative_gap_events(const GapDataset *dataset, size_t max_states)
{
    size_t samples = rule_search_sample_count(dataset, max_states);
    size_t events = 0;
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t true_gap = dataset->gap[index];
        if (true_gap > 2u) {
            events += (size_t)(true_gap / 2u - 1u);
        }
    }
    return events;
}

static uint64_t rule_coverage_cache_mix(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    hash *= UINT64_C(1099511628211);
    return hash;
}

static uint64_t rule_coverage_cache_key(const RuleSearchConfig *config,
                                        const RuleSearchGrammar *grammar,
                                        const ExpandedRuleSet *rules,
                                        const GapDataset *dataset,
                                        size_t max_states,
                                        size_t samples,
                                        size_t events)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = rule_coverage_cache_mix(hash, (uint64_t)RULE_SEARCH_GRAMMAR_VERSION);
    hash = rule_coverage_cache_mix(hash, (uint64_t)config->train_limit);
    hash = rule_coverage_cache_mix(hash, (uint64_t)config->test_limit);
    hash = rule_coverage_cache_mix(hash, (uint64_t)max_states);
    hash = rule_coverage_cache_mix(hash, (uint64_t)samples);
    hash = rule_coverage_cache_mix(hash, (uint64_t)events);
    hash = rule_coverage_cache_mix(hash, (uint64_t)config->include_state_conditions);
    hash = rule_coverage_cache_mix(hash, (uint64_t)config->include_candidate_filters);
    hash = rule_coverage_cache_mix(hash, (uint64_t)config->include_gap_filters);
    hash = rule_coverage_cache_mix(hash, (uint64_t)config->mod_count);
    hash = rule_coverage_cache_mix(hash, (uint64_t)grammar->state_count);
    hash = rule_coverage_cache_mix(hash, (uint64_t)grammar->filter_count);
    hash = rule_coverage_cache_mix(hash, (uint64_t)rules->count);
    for (size_t i = 0; i < config->mod_count; ++i) {
        hash = rule_coverage_cache_mix(hash, (uint64_t)config->moduli[i]);
        hash = rule_coverage_cache_mix(hash, (uint64_t)grammar->state_offsets[i]);
        hash = rule_coverage_cache_mix(hash, (uint64_t)grammar->candidate_offsets[i]);
        hash = rule_coverage_cache_mix(hash, (uint64_t)grammar->gap_offsets[i]);
    }
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        hash = rule_coverage_cache_mix(hash, (uint64_t)dataset->prime[index]);
        hash = rule_coverage_cache_mix(hash, (uint64_t)dataset->gap[index]);
    }
    for (size_t i = 0; i < rules->count; ++i) {
        const ExpandedRule *rule = &rules->items[i];
        hash = rule_coverage_cache_mix(hash, (uint64_t)rule->kind);
        hash = rule_coverage_cache_mix(hash, (uint64_t)rule->state_modulus);
        hash = rule_coverage_cache_mix(hash, (uint64_t)rule->state_residue);
        hash = rule_coverage_cache_mix(hash, (uint64_t)rule->filter_modulus);
        hash = rule_coverage_cache_mix(hash, (uint64_t)rule->filter_residue);
    }
    return hash;
}

static int rule_coverage_cache_path(const RuleSearchConfig *config,
                                    uint64_t key,
                                    char *path,
                                    size_t capacity)
{
    if (config == NULL || config->coverage_cache_dir == NULL ||
        config->coverage_cache_dir[0] == '\0' || path == NULL || capacity == 0) {
        return 0;
    }
    size_t dir_len = strlen(config->coverage_cache_dir);
    const char *separator =
        dir_len != 0 && config->coverage_cache_dir[dir_len - 1u] == '/' ? "" : "/";
    int written = snprintf(path,
                           capacity,
                           "%s%sprime-rule-coverage-%016llx.bin",
                           config->coverage_cache_dir,
                           separator,
                           (unsigned long long)key);
    return written > 0 && (size_t)written < capacity;
}

static int ensure_rule_coverage_cache_dir(const RuleSearchConfig *config)
{
    if (config == NULL || config->coverage_cache_dir == NULL ||
        config->coverage_cache_dir[0] == '\0') {
        return 0;
    }
    if (mkdir(config->coverage_cache_dir, 0777) == 0 || errno == EEXIST) {
        return 1;
    }
    return 0;
}

static int load_rule_coverage_cache(const char *path,
                                    uint64_t key,
                                    size_t expected_rule_count,
                                    size_t expected_event_count,
                                    size_t expected_samples,
                                    NNRuleCoverage *coverage)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return 0;
    }

    RuleCoverageCacheHeader header;
    int ok = fread(&header, sizeof(header), 1u, fp) == 1u &&
             memcmp(header.magic, "crlp.cover.v1", strlen("crlp.cover.v1")) == 0 &&
             header.key == key &&
             header.rule_count == (uint64_t)expected_rule_count &&
             header.event_count == (uint64_t)expected_event_count &&
             header.samples == (uint64_t)expected_samples;
    if (!ok || header.word_count != (uint64_t)((expected_event_count + 63u) / 64u)) {
        fclose(fp);
        return 0;
    }

    if (!nn_rule_coverage_init(coverage,
                               (size_t)header.rule_count,
                               (size_t)header.event_count)) {
        fclose(fp);
        return 0;
    }
    size_t words = (size_t)header.rule_count * (size_t)header.word_count;
    if (words != 0 &&
        fread(coverage->bits, sizeof(*coverage->bits), words, fp) != words) {
        nn_rule_coverage_free(coverage);
        fclose(fp);
        return 0;
    }
    if (fclose(fp) != 0) {
        nn_rule_coverage_free(coverage);
        return 0;
    }
    return 1;
}

static int save_rule_coverage_cache(const char *path,
                                    uint64_t key,
                                    size_t samples,
                                    const NNRuleCoverage *coverage)
{
    if (path == NULL || coverage == NULL) {
        return 0;
    }

    char tmp_path[1024];
    int written = snprintf(tmp_path,
                           sizeof(tmp_path),
                           "%s.tmp.%llu.%llu",
                           path,
                           (unsigned long long)getpid(),
                           (unsigned long long)time(NULL));
    if (written <= 0 || (size_t)written >= sizeof(tmp_path)) {
        return 0;
    }

    FILE *fp = fopen(tmp_path, "wb");
    if (fp == NULL) {
        return 0;
    }

    RuleCoverageCacheHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "crlp.cover.v1", strlen("crlp.cover.v1"));
    header.key = key;
    header.rule_count = (uint64_t)coverage->rule_count;
    header.event_count = (uint64_t)coverage->event_count;
    header.word_count = (uint64_t)coverage->word_count;
    header.samples = (uint64_t)samples;

    size_t words = coverage->rule_count * coverage->word_count;
    int ok = fwrite(&header, sizeof(header), 1u, fp) == 1u &&
             (words == 0 ||
              fwrite(coverage->bits, sizeof(*coverage->bits), words, fp) == words);
    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || rename(tmp_path, path) != 0) {
        remove(tmp_path);
        return 0;
    }
    return 1;
}

static void mark_matching_rule_coverage(NNRuleCoverage *coverage,
                                        const RuleSearchConfig *config,
                                        const RuleSearchGrammar *grammar,
                                        uint32_t p,
                                        uint32_t gap,
                                        size_t event_index)
{
    size_t state_indices[RULE_SEARCH_MAX_MODULI + 1u];
    size_t filter_indices[RULE_SEARCH_MAX_MODULI * 2u];
    size_t state_count = 1;
    size_t filter_count = 0;
    uint32_t candidate = p + gap;

    state_indices[0] = 0;
    for (size_t i = 0; i < config->mod_count; ++i) {
        uint32_t modulus = config->moduli[i];
        if (grammar->state_offsets[i] != (size_t)-1) {
            state_indices[state_count++] = grammar->state_offsets[i] + (p % modulus);
        }
        if (candidate > modulus) {
            if (grammar->candidate_offsets[i] != (size_t)-1) {
                filter_indices[filter_count++] =
                    grammar->candidate_offsets[i] + (candidate % modulus);
            }
            if (grammar->gap_offsets[i] != (size_t)-1) {
                filter_indices[filter_count++] = grammar->gap_offsets[i] + (gap % modulus);
            }
        }
    }

    for (size_t state = 0; state < state_count; ++state) {
        size_t base = state_indices[state] * grammar->filter_count;
        for (size_t filter = 0; filter < filter_count; ++filter) {
            size_t rule_index = base + filter_indices[filter];
            nn_rule_coverage_set(coverage, rule_index, event_index);
        }
    }
}

static int build_rule_coverage(const RuleSearchConfig *config,
                               const RuleSearchGrammar *grammar,
                               const ExpandedRuleSet *rules,
                               const GapDataset *dataset,
                               size_t max_states,
                               NNRuleCoverage *coverage,
                               size_t *samples_out)
{
    memset(coverage, 0, sizeof(*coverage));
    size_t samples = rule_search_sample_count(dataset, max_states);
    size_t events = count_negative_gap_events(dataset, max_states);
    if (samples_out != NULL) {
        *samples_out = samples;
    }
    char cache_path[1024];
    uint64_t cache_key = rule_coverage_cache_key(config,
                                                 grammar,
                                                 rules,
                                                 dataset,
                                                 max_states,
                                                 samples,
                                                 events);
    int use_cache = rule_coverage_cache_path(config,
                                             cache_key,
                                             cache_path,
                                             sizeof(cache_path));
    if (use_cache &&
        load_rule_coverage_cache(cache_path,
                                 cache_key,
                                 rules->count,
                                 events,
                                 samples,
                                 coverage)) {
        ++RULE_COVERAGE_CACHE_STATS.hits;
        return 1;
    }
    if (use_cache) {
        ++RULE_COVERAGE_CACHE_STATS.misses;
    }

    memset(coverage, 0, sizeof(*coverage));
    if (!nn_rule_coverage_init(coverage, rules->count, events)) {
        return 0;
    }
    if (events == 0 || rules->count == 0) {
        if (use_cache && ensure_rule_coverage_cache_dir(config) &&
            save_rule_coverage_cache(cache_path, cache_key, samples, coverage)) {
            ++RULE_COVERAGE_CACHE_STATS.writes;
        }
        return 1;
    }

    size_t event_index = 0;
    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t true_gap = dataset->gap[index];
        for (uint32_t gap = 2; gap < true_gap; gap += 2) {
            mark_matching_rule_coverage(coverage,
                                        config,
                                        grammar,
                                        p,
                                        gap,
                                        event_index++);
        }
    }
    if (event_index != events) {
        nn_rule_coverage_free(coverage);
        return 0;
    }
    if (use_cache && ensure_rule_coverage_cache_dir(config) &&
        save_rule_coverage_cache(cache_path, cache_key, samples, coverage)) {
        ++RULE_COVERAGE_CACHE_STATS.writes;
    }
    return 1;
}

static int rule_is_marginal_candidate(const ExpandedRule *rule, int validation)
{
    if (!expanded_rule_has_deterministic_certificate(rule)) {
        return 0;
    }
    if (validation) {
        return rule->validation_false_rejections == 0 && rule->validation_rejects != 0;
    }
    return rule->train_false_rejections == 0 && rule->train_rejects != 0;
}

PRIME_GAP_TEST_API int compute_marginal_rule_utilities(ExpandedRuleSet *rules,
                                                       const RuleSearchConfig *config,
                                                       const RuleSearchGrammar *grammar,
                                                       const GapDataset *dataset,
                                                       size_t max_states,
                                                       int validation,
                                                       size_t *samples_out,
                                                       size_t *events_out)
{
    NNRuleCoverage coverage;
    size_t samples = 0;
    if (!build_rule_coverage(config, grammar, rules, dataset, max_states, &coverage, &samples)) {
        return 0;
    }

    if (samples_out != NULL) {
        *samples_out = samples;
    }
    if (events_out != NULL) {
        *events_out = coverage.event_count;
    }
    for (size_t i = 0; i < rules->count; ++i) {
        if (validation) {
            rules->items[i].validation_marginal_utility = 0.0;
        } else {
            rules->items[i].train_marginal_utility = 0.0;
        }
    }
    if (coverage.event_count == 0 || coverage.word_count == 0) {
        nn_rule_coverage_free(&coverage);
        return 1;
    }

    uint8_t *selected = calloc(rules->count, sizeof(*selected));
    uint64_t *covered = calloc(coverage.word_count, sizeof(*covered));
    if (selected == NULL || covered == NULL) {
        free(selected);
        free(covered);
        nn_rule_coverage_free(&coverage);
        return 0;
    }

    for (size_t rank = 0; rank < config->max_rules; ++rank) {
        size_t best_index = (size_t)-1;
        size_t best_gain = 0;
        for (size_t i = 0; i < rules->count; ++i) {
            if (selected[i] || !rule_is_marginal_candidate(&rules->items[i], validation)) {
                continue;
            }
            size_t gain = nn_rule_coverage_gain(&coverage, i, covered);
            if (gain > best_gain) {
                best_gain = gain;
                best_index = i;
            }
        }
        if (best_index == (size_t)-1 || best_gain == 0) {
            break;
        }

        selected[best_index] = 1;
        if (validation) {
            rules->items[best_index].validation_marginal_utility = (double)best_gain;
        } else {
            rules->items[best_index].train_marginal_utility = (double)best_gain;
        }
        nn_rule_coverage_add_rule_to_residual(&coverage, best_index, covered);
    }

    free(selected);
    free(covered);
    nn_rule_coverage_free(&coverage);
    return 1;
}

PRIME_GAP_TEST_API void fill_expanded_rule_features(const ExpandedRule *rule,
                                                    uint32_t max_modulus,
                                                    nn_real *features)
{
    double max_mod = max_modulus == 0 ? 1.0 : (double)max_modulus;
    uint32_t state_common = expanded_rule_is_conditional(rule)
                                ? gcd_u32(rule->state_modulus, rule->state_residue)
                                : 0u;
    uint32_t filter_common = gcd_u32(rule->filter_modulus, rule->filter_residue);
    uint32_t candidate_residue = expanded_rule_structural_candidate_residue(rule);
    uint32_t candidate_common = gcd_u32(rule->filter_modulus, candidate_residue);

    features[0] = expanded_rule_is_conditional(rule) ? (nn_real)1 : (nn_real)0;
    features[1] = expanded_rule_is_conditional(rule)
                      ? (nn_real)((double)rule->state_modulus / max_mod)
                      : (nn_real)0;
    features[2] = expanded_rule_is_conditional(rule)
                      ? (nn_real)((double)rule->state_residue /
                                  (double)rule->state_modulus)
                      : (nn_real)0;
    features[3] = expanded_rule_is_conditional(rule) && rule->state_residue == 0
                      ? (nn_real)1
                      : (nn_real)0;
    features[4] = expanded_rule_is_conditional(rule)
                      ? (nn_real)((double)state_common / (double)rule->state_modulus)
                      : (nn_real)0;
    features[5] = rule->kind == RULE_FILTER_GAP_MOD ? (nn_real)1 : (nn_real)0;
    features[6] = (nn_real)((double)rule->filter_modulus / max_mod);
    features[7] = (nn_real)((double)rule->filter_residue /
                            (double)rule->filter_modulus);
    features[8] = rule->filter_residue == 0 ? (nn_real)1 : (nn_real)0;
    features[9] = (nn_real)((double)filter_common / (double)rule->filter_modulus);
    features[10] = rule->state_modulus == rule->filter_modulus ? (nn_real)1 : (nn_real)0;
    features[11] = (nn_real)((double)candidate_residue /
                             (double)rule->filter_modulus);
    features[12] = (nn_real)((double)candidate_common / (double)rule->filter_modulus);
    features[13] = (nn_real)(1.0 / (double)rule->filter_modulus);
    features[14] = expanded_rule_is_conditional(rule)
                       ? (nn_real)(1.0 / (double)rule->state_modulus)
                       : (nn_real)0;
}

PRIME_GAP_TEST_API void set_expanded_rule_utilities(ExpandedRuleSet *rules)
{
    for (size_t i = 0; i < rules->count; ++i) {
        ExpandedRule *rule = &rules->items[i];
        rule->utility = rule->validation_false_rejections == 0
                            ? rule->validation_marginal_utility
                            : -(double)rule->validation_false_rejections;
    }
}

PRIME_GAP_TEST_API void fill_rule_proposer_sample(const ExpandedRule *rule,
                                                  uint32_t max_modulus,
                                                  double target_scale,
                                                  RuleProposerSample *sample)
{
    memset(sample, 0, sizeof(*sample));
    fill_expanded_rule_features(rule, max_modulus, sample->inputs.values);
    sample->targets.train_rejects = rule->train_rejects;
    sample->targets.train_false_rejections = rule->train_false_rejections;
    sample->targets.train_marginal_utility = rule->train_marginal_utility;
    sample->targets.marginal_target =
        rule->train_false_rejections == 0
            ? (nn_real)(rule->train_marginal_utility / target_scale)
            : (nn_real)0;
}

static int score_expanded_rules_with_model(ExpandedRuleSet *rules,
                                           const RuleSearchConfig *config,
                                           double *loss_out,
                                           size_t *parameters_out,
                                           size_t *forward_flops_out)
{
    double max_train_marginal_utility = 0.0;
    uint32_t max_modulus = 0;
    for (size_t i = 0; i < config->mod_count; ++i) {
        if (config->moduli[i] > max_modulus) {
            max_modulus = config->moduli[i];
        }
    }
    for (size_t i = 0; i < rules->count; ++i) {
        if (rules->items[i].train_marginal_utility > max_train_marginal_utility) {
            max_train_marginal_utility = rules->items[i].train_marginal_utility;
        }
    }

    RuleProposerSample *samples = calloc(rules->count, sizeof(*samples));
    nn_real *all_inputs = calloc(rules->count * RULE_SEARCH_FEATURE_COUNT, sizeof(*all_inputs));
    nn_real *train_inputs = calloc(rules->count * RULE_SEARCH_FEATURE_COUNT, sizeof(*train_inputs));
    nn_real *train_targets = calloc(rules->count, sizeof(*train_targets));
    nn_real *scores = calloc(rules->count, sizeof(*scores));
    if (samples == NULL || all_inputs == NULL || train_inputs == NULL ||
        train_targets == NULL || scores == NULL) {
        free(samples);
        free(all_inputs);
        free(train_inputs);
        free(train_targets);
        free(scores);
        return 0;
    }

    double target_scale = max_train_marginal_utility <= 0.0
                              ? 1.0
                              : max_train_marginal_utility;
    size_t positive_count = 0;
    size_t zero_eligible_count = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        fill_rule_proposer_sample(&rules->items[i], max_modulus, target_scale, &samples[i]);
        memcpy(&all_inputs[i * RULE_SEARCH_FEATURE_COUNT],
               samples[i].inputs.values,
               RULE_SEARCH_FEATURE_COUNT * sizeof(*all_inputs));
        if (samples[i].targets.marginal_target > (nn_real)0) {
            ++positive_count;
        } else if (rule_is_marginal_candidate(&rules->items[i], 0)) {
            ++zero_eligible_count;
        }
    }

    size_t negative_limit = positive_count == 0 ? zero_eligible_count : positive_count * 8u;
    if (negative_limit > zero_eligible_count) {
        negative_limit = zero_eligible_count;
    }
    size_t zero_seen = 0;
    size_t zero_kept = 0;
    size_t train_count = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        int keep = samples[i].targets.marginal_target > (nn_real)0;
        if (!keep && rule_is_marginal_candidate(&rules->items[i], 0) &&
            negative_limit != 0 && zero_eligible_count != 0) {
            size_t before = zero_seen * negative_limit / zero_eligible_count;
            ++zero_seen;
            size_t after = zero_seen * negative_limit / zero_eligible_count;
            keep = after > before && zero_kept < negative_limit;
            if (keep) {
                ++zero_kept;
            }
        }
        if (!keep) {
            continue;
        }
        memcpy(&train_inputs[train_count * RULE_SEARCH_FEATURE_COUNT],
               samples[i].inputs.values,
               RULE_SEARCH_FEATURE_COUNT * sizeof(*train_inputs));
        train_targets[train_count] = samples[i].targets.marginal_target;
        ++train_count;
    }
    if (train_count == 0) {
        for (size_t i = 0; i < rules->count; ++i) {
            memcpy(&train_inputs[i * RULE_SEARCH_FEATURE_COUNT],
                   samples[i].inputs.values,
                   RULE_SEARCH_FEATURE_COUNT * sizeof(*train_inputs));
            train_targets[i] = samples[i].targets.marginal_target;
        }
        train_count = rules->count;
    }
    set_expanded_rule_utilities(rules);

    const size_t layers[] = {RULE_SEARCH_FEATURE_COUNT, 20, 10, 1};
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    nn_config.seed = 92831;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&nn_config);
    if (net == NULL) {
        free(samples);
        free(all_inputs);
        free(train_inputs);
        free(train_targets);
        free(scores);
        return 0;
    }

    nn_real loss = nn_train_batched(net,
                                    train_inputs,
                                    train_targets,
                                    train_count,
                                    256,
                                    (nn_real)0.02,
                                    RULE_SEARCH_SCORER_EPOCHS,
                                    1);
    nn_forward_batch(net, all_inputs, scores, rules->count);
    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].model_score = scores[i];
    }
    if (loss_out != NULL) {
        *loss_out = (double)loss;
    }
    if (parameters_out != NULL) {
        *parameters_out = nn_parameter_count(net);
    }
    if (forward_flops_out != NULL) {
        *forward_flops_out = nn_forward_flop_count(net);
    }

    nn_free(net);
    free(samples);
    free(all_inputs);
    free(train_inputs);
    free(train_targets);
    free(scores);
    return 1;
}

static uint64_t mix_rule_hash_u64(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    hash *= UINT64_C(1099511628211);
    return hash;
}

static uint64_t expanded_rule_key_hash(const ExpandedRule *rule, uint64_t seed)
{
    uint64_t hash = UINT64_C(1469598103934665603) ^ seed;
    hash = mix_rule_hash_u64(hash, (uint64_t)rule->kind);
    hash = mix_rule_hash_u64(hash, (uint64_t)rule->state_modulus);
    hash = mix_rule_hash_u64(hash, (uint64_t)rule->state_residue);
    hash = mix_rule_hash_u64(hash, (uint64_t)rule->filter_modulus);
    hash = mix_rule_hash_u64(hash, (uint64_t)rule->filter_residue);
    return hash;
}

static double random_score_from_hash(uint64_t hash)
{
    return (double)(hash >> 11) * (1.0 / 9007199254740992.0);
}

static double structural_rule_score(const ExpandedRule *rule)
{
    uint32_t candidate_residue = expanded_rule_structural_candidate_residue(rule);
    uint32_t candidate_common = gcd_u32(rule->filter_modulus, candidate_residue);
    double modulus = (double)rule->filter_modulus;
    double gcd_ratio = modulus == 0.0 ? 0.0 : (double)candidate_common / modulus;
    double support_prior = modulus == 0.0 ? 0.0 : 1.0 / modulus;
    double score = 0.0;

    if (expanded_rule_has_deterministic_certificate(rule)) {
        score += 4.0;
    }
    if (!expanded_rule_is_conditional(rule)) {
        score += 1.0;
    }
    if (rule->kind == RULE_FILTER_CANDIDATE_MOD) {
        score += 0.5;
    }
    if (candidate_residue == 0u) {
        score += 0.25;
    }
    score += gcd_ratio;
    score += support_prior;
    return score;
}

static int score_expanded_rules_without_model(ExpandedRuleSet *rules,
                                              const RuleSearchConfig *config,
                                              double *loss_out)
{
    set_expanded_rule_utilities(rules);
    for (size_t i = 0; i < rules->count; ++i) {
        ExpandedRule *rule = &rules->items[i];
        switch (config->proposer_mode) {
        case RULE_PROPOSER_RANDOM:
            rule->model_score =
                (nn_real)random_score_from_hash(expanded_rule_key_hash(rule,
                                                                       config->proposer_seed));
            break;
        case RULE_PROPOSER_GRAMMAR:
            rule->model_score = (nn_real)(rules->count - i);
            break;
        case RULE_PROPOSER_STRUCTURAL:
            rule->model_score = (nn_real)structural_rule_score(rule);
            break;
        case RULE_PROPOSER_VALIDATION_ORACLE:
            rule->model_score = (nn_real)rule->utility;
            break;
        case RULE_PROPOSER_NEURAL:
        case RULE_PROPOSER_CLOSED_LOOP:
        case RULE_PROPOSER_UNIFIED:
            return 0;
        }
    }
    if (loss_out != NULL) {
        *loss_out = 0.0;
    }
    return 1;
}

static int score_expanded_rules(ExpandedRuleSet *rules,
                                const RuleSearchConfig *config,
                                RuleSearchMetrics *metrics)
{
    double *loss_out = metrics == NULL ? NULL : &metrics->proposer_loss;
    if (config->proposer_mode == RULE_PROPOSER_NEURAL) {
        return score_expanded_rules_with_model(rules,
                                               config,
                                               loss_out,
                                               metrics == NULL ? NULL : &metrics->parameters,
                                               metrics == NULL ? NULL : &metrics->forward_flops);
    }
    return score_expanded_rules_without_model(rules, config, loss_out);
}

static int compare_expanded_rules_model(const void *left, const void *right)
{
    const ExpandedRule *a = (const ExpandedRule *)left;
    const ExpandedRule *b = (const ExpandedRule *)right;

    if ((double)a->model_score < (double)b->model_score) {
        return 1;
    }
    if ((double)a->model_score > (double)b->model_score) {
        return -1;
    }
    if (a->state_modulus != b->state_modulus) {
        return a->state_modulus < b->state_modulus ? -1 : 1;
    }
    if (a->state_residue != b->state_residue) {
        return a->state_residue < b->state_residue ? -1 : 1;
    }
    if (a->kind != b->kind) {
        return a->kind < b->kind ? -1 : 1;
    }
    if (a->filter_modulus != b->filter_modulus) {
        return a->filter_modulus < b->filter_modulus ? -1 : 1;
    }
    if (a->filter_residue != b->filter_residue) {
        return a->filter_residue < b->filter_residue ? -1 : 1;
    }
    return 0;
}

static int expanded_candidate_rule_subset_of(const ExpandedRule *candidate,
                                             const ExpandedRule *accepted)
{
    return candidate->kind == RULE_FILTER_CANDIDATE_MOD &&
           accepted->kind == RULE_FILTER_CANDIDATE_MOD &&
           candidate->state_modulus == 1u &&
           accepted->state_modulus == 1u &&
           candidate->filter_modulus % accepted->filter_modulus == 0u &&
           candidate->filter_residue % accepted->filter_modulus ==
               accepted->filter_residue;
}

static int expanded_rule_same_key(const ExpandedRule *a, const ExpandedRule *b)
{
    return a->kind == b->kind &&
           a->state_modulus == b->state_modulus &&
           a->state_residue == b->state_residue &&
           a->filter_modulus == b->filter_modulus &&
           a->filter_residue == b->filter_residue;
}

static int expanded_rule_is_covered(const ExpandedRuleSet *accepted,
                                    const ExpandedRule *candidate)
{
    for (size_t i = 0; i < accepted->count; ++i) {
        if (expanded_candidate_rule_subset_of(candidate, &accepted->items[i])) {
            return 1;
        }
    }
    return 0;
}

static int expanded_rule_is_covered_by_indices(const ExpandedRuleSet *rules,
                                               const size_t *accepted_indices,
                                               size_t accepted_count,
                                               const ExpandedRule *candidate)
{
    if (rules == NULL || accepted_indices == NULL) {
        return 0;
    }
    for (size_t i = 0; i < accepted_count; ++i) {
        size_t accepted_index = accepted_indices[i];
        if (accepted_index < rules->count &&
            expanded_candidate_rule_subset_of(candidate, &rules->items[accepted_index])) {
            return 1;
        }
    }
    return 0;
}

static void expanded_rule_remove_subsets(ExpandedRuleSet *accepted,
                                         const ExpandedRule *candidate)
{
    size_t write = 0;
    for (size_t read = 0; read < accepted->count; ++read) {
        if (expanded_candidate_rule_subset_of(&accepted->items[read], candidate)) {
            continue;
        }
        if (write != read) {
            accepted->items[write] = accepted->items[read];
        }
        ++write;
    }
    accepted->count = write;
}

static int expanded_rule_is_selectable(const ExpandedRule *rule)
{
    return rule->train_false_rejections == 0 &&
           rule->validation_false_rejections == 0 &&
           (rule->train_rejects != 0 || rule->validation_rejects != 0) &&
           expanded_rule_has_deterministic_certificate(rule);
}

static void summarize_expanded_rule_candidates(const ExpandedRuleSet *rules,
                                               RuleSearchMetrics *metrics)
{
    metrics->proposed_rules = rules->count;
    for (size_t i = 0; i < rules->count; ++i) {
        const ExpandedRule *rule = &rules->items[i];
        if (rule->train_false_rejections != 0 || rule->validation_false_rejections != 0) {
            ++metrics->unsafe_rules;
            continue;
        }
        if (rule->train_rejects == 0 && rule->validation_rejects == 0) {
            ++metrics->no_support_rules;
            continue;
        }
        ++metrics->verified_rules;
        if (!expanded_rule_has_deterministic_certificate(rule)) {
            ++metrics->uncertified_rules;
        }
    }
}

static void mark_expanded_rule_acceptance(ExpandedRuleSet *rules,
                                          const ExpandedRuleSet *accepted,
                                          RuleSearchMetrics *metrics)
{
    metrics->accepted_rules = accepted->count;
    metrics->accepted_conditional_rules = 0;
    metrics->accepted_gap_rules = 0;
    metrics->accepted_candidate_rules = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].accepted = 0;
    }
    for (size_t i = 0; i < accepted->count; ++i) {
        if (expanded_rule_is_conditional(&accepted->items[i])) {
            ++metrics->accepted_conditional_rules;
        }
        if (accepted->items[i].kind == RULE_FILTER_GAP_MOD) {
            ++metrics->accepted_gap_rules;
        } else {
            ++metrics->accepted_candidate_rules;
        }
        for (size_t candidate = 0; candidate < rules->count; ++candidate) {
            if (expanded_rule_same_key(&rules->items[candidate], &accepted->items[i])) {
                rules->items[candidate].accepted = 1;
                break;
            }
        }
    }
}

typedef struct {
    const ExpandedRuleSet *rules;
    const size_t *accepted_indices;
    size_t accepted_count;
    size_t accepted_candidate_rules;
    size_t accepted_gap_rules;
    size_t accepted_conditional_rules;
    size_t event_count;
    size_t covered_events;
    size_t max_rules;
} RuleDiscoveryState;

static RuleDiscoveryState build_rule_discovery_state(const ExpandedRuleSet *rules,
                                                     const size_t *accepted_indices,
                                                     size_t accepted_count,
                                                     const NNRuleCoverage *coverage,
                                                     const uint64_t *covered,
                                                     size_t max_rules)
{
    RuleDiscoveryState state;
    memset(&state, 0, sizeof(state));
    state.rules = rules;
    state.accepted_indices = accepted_indices;
    state.accepted_count = accepted_count;
    state.event_count = coverage->event_count;
    state.covered_events = nn_rule_coverage_count_covered(covered, coverage->word_count);
    state.max_rules = max_rules;
    for (size_t i = 0; i < accepted_count; ++i) {
        size_t accepted_index = accepted_indices[i];
        if (rules == NULL || accepted_index >= rules->count) {
            continue;
        }
        const ExpandedRule *accepted = &rules->items[accepted_index];
        if (expanded_rule_is_conditional(accepted)) {
            ++state.accepted_conditional_rules;
        }
        if (accepted->kind == RULE_FILTER_GAP_MOD) {
            ++state.accepted_gap_rules;
        } else {
            ++state.accepted_candidate_rules;
        }
    }
    return state;
}

static void fill_rule_discovery_features(const ExpandedRule *rule,
                                         uint32_t max_modulus,
                                         const RuleDiscoveryState *state,
                                         nn_real *features)
{
    fill_expanded_rule_features(rule, max_modulus, features);

    double max_rules = state->max_rules == 0 ? 1.0 : (double)state->max_rules;
    double accepted_count = (double)state->accepted_count;
    double accepted_scale = accepted_count <= 0.0 ? 1.0 : accepted_count;
    double residual_events = state->event_count <= state->covered_events
                                 ? 0.0
                                 : (double)(state->event_count - state->covered_events);
    double residual_fraction = state->event_count == 0
                                   ? 0.0
                                   : residual_events / (double)state->event_count;
    uint32_t candidate_residue = expanded_rule_structural_candidate_residue(rule);
    size_t same_kind = 0;
    size_t same_filter_modulus = 0;
    size_t same_state_modulus = 0;
    size_t same_candidate_residue = 0;

    if (state->rules != NULL && state->accepted_indices != NULL) {
        for (size_t i = 0; i < state->accepted_count; ++i) {
            size_t accepted_index = state->accepted_indices[i];
            if (accepted_index >= state->rules->count) {
                continue;
            }
            const ExpandedRule *accepted = &state->rules->items[accepted_index];
            if (accepted->kind == rule->kind) {
                ++same_kind;
            }
            if (accepted->filter_modulus == rule->filter_modulus) {
                ++same_filter_modulus;
            }
            if (accepted->state_modulus == rule->state_modulus) {
                ++same_state_modulus;
            }
            if (expanded_rule_structural_candidate_residue(accepted) == candidate_residue) {
                ++same_candidate_residue;
            }
        }
    }

    features[RULE_SEARCH_FEATURE_COUNT + 0] = (nn_real)(accepted_count / max_rules);
    features[RULE_SEARCH_FEATURE_COUNT + 1] = (nn_real)residual_fraction;
    features[RULE_SEARCH_FEATURE_COUNT + 2] =
        (nn_real)((double)state->accepted_candidate_rules / max_rules);
    features[RULE_SEARCH_FEATURE_COUNT + 3] =
        (nn_real)((double)state->accepted_gap_rules / max_rules);
    features[RULE_SEARCH_FEATURE_COUNT + 4] =
        (nn_real)((double)state->accepted_conditional_rules / max_rules);
    features[RULE_SEARCH_FEATURE_COUNT + 5] = (nn_real)((double)same_kind / accepted_scale);
    features[RULE_SEARCH_FEATURE_COUNT + 6] =
        (nn_real)((double)same_filter_modulus / accepted_scale);
    features[RULE_SEARCH_FEATURE_COUNT + 7] =
        (nn_real)((double)same_state_modulus / accepted_scale);
    features[RULE_SEARCH_FEATURE_COUNT + 8] =
        (nn_real)((double)same_candidate_residue / accepted_scale);
    features[RULE_SEARCH_FEATURE_COUNT + 9] =
        state->rules != NULL &&
                expanded_rule_is_covered_by_indices(state->rules,
                                                    state->accepted_indices,
                                                    state->accepted_count,
                                                    rule)
            ? (nn_real)1
            : (nn_real)0;
}

static uint32_t max_rule_search_modulus(const RuleSearchConfig *config)
{
    uint32_t max_modulus = 0;
    for (size_t i = 0; i < config->mod_count; ++i) {
        if (config->moduli[i] > max_modulus) {
            max_modulus = config->moduli[i];
        }
    }
    return max_modulus;
}

typedef struct {
    const ExpandedRuleSet *rules;
    const RuleSearchConfig *config;
    uint32_t max_modulus;
} PrimeClosedLoopContext;

static int prime_closed_loop_candidate_allowed(size_t rule_index,
                                               const size_t *accepted_indices,
                                               size_t accepted_count,
                                               void *context)
{
    PrimeClosedLoopContext *prime = (PrimeClosedLoopContext *)context;
    if (prime == NULL || prime->rules == NULL || rule_index >= prime->rules->count) {
        return 0;
    }
    const ExpandedRule *rule = &prime->rules->items[rule_index];
    return expanded_rule_is_selectable(rule) &&
           !expanded_rule_is_covered_by_indices(prime->rules,
                                                accepted_indices,
                                                accepted_count,
                                                rule);
}

static int prime_closed_loop_keep_accepted(size_t accepted_rule_index,
                                           size_t candidate_rule_index,
                                           void *context)
{
    PrimeClosedLoopContext *prime = (PrimeClosedLoopContext *)context;
    if (prime == NULL || prime->rules == NULL ||
        accepted_rule_index >= prime->rules->count ||
        candidate_rule_index >= prime->rules->count) {
        return 0;
    }
    return !expanded_candidate_rule_subset_of(&prime->rules->items[accepted_rule_index],
                                              &prime->rules->items[candidate_rule_index]);
}

static int prime_closed_loop_encode_features(size_t rule_index,
                                             const size_t *accepted_indices,
                                             size_t accepted_count,
                                             const NNRuleCoverage *coverage,
                                             const uint64_t *covered,
                                             void *context,
                                             nn_real *features,
                                             size_t capacity)
{
    PrimeClosedLoopContext *prime = (PrimeClosedLoopContext *)context;
    if (prime == NULL || prime->rules == NULL || coverage == NULL || covered == NULL ||
        features == NULL || capacity < (size_t)RULE_DISCOVERY_FEATURE_COUNT ||
        rule_index >= prime->rules->count) {
        return 0;
    }
    RuleDiscoveryState state =
        build_rule_discovery_state(prime->rules,
                                   accepted_indices,
                                   accepted_count,
                                   coverage,
                                   covered,
                                   prime->config == NULL ? 0 : prime->config->max_rules);
    fill_rule_discovery_features(&prime->rules->items[rule_index],
                                 prime->max_modulus,
                                 &state,
                                 features);
    return 1;
}

static void prime_unified_feature_schema(NNFeatureSchema *schema, char names[UNIFIED_SCORER_FEATURE_COUNT][64])
{
    for (size_t i = 0; i < (size_t)UNIFIED_SCORER_FEATURE_COUNT; ++i) {
        const char *slot_name = nn_scorer_envelope_slot_name(i);
        snprintf(names[i],
                 64,
                 "%s",
                 slot_name == NULL ? "unified_padding" : slot_name);
        if (slot_name == NULL) {
            snprintf(names[i], 64, "unified_padding_%02zu", i);
        }
        schema[i].name = names[i];
        schema[i].provenance = NN_FEATURE_STRUCTURAL_PRIOR;
        schema[i].enabled_by_default = 0;
    }

    schema[UNIFIED_F_BIAS].enabled_by_default = 1;
    schema[UNIFIED_F_ROLE_RULE].enabled_by_default = 1;
    schema[UNIFIED_F_DOMAIN_PRIME_RULE].enabled_by_default = 1;
    schema[UNIFIED_F_ROLE_ACTION].enabled_by_default = 1;
    schema[UNIFIED_F_DOMAIN_PRIME_ACTION].enabled_by_default = 1;
    schema[UNIFIED_F_RULE_INDEX].enabled_by_default = 1;
    schema[UNIFIED_F_CANDIDATE_COUNT].enabled_by_default = 1;
    schema[UNIFIED_F_DOMAIN_GRID_ACTION].enabled_by_default = 1;
    schema[UNIFIED_F_DOMAIN_CLAIM_ACTION].enabled_by_default = 1;
    schema[UNIFIED_F_DOMAIN_ARC_ACTION].enabled_by_default = 1;

    for (size_t i = 0; i < (size_t)RULE_DISCOVERY_FEATURE_COUNT; ++i) {
        size_t slot = (size_t)UNIFIED_F_RULE_FEATURES + i;
        snprintf(names[slot], 64, "prime_rule_%s", PRIME_CLOSED_LOOP_RULE_FEATURES[i].name);
        schema[slot].provenance = PRIME_CLOSED_LOOP_RULE_FEATURES[i].provenance;
        schema[slot].enabled_by_default = PRIME_CLOSED_LOOP_RULE_FEATURES[i].enabled_by_default;
    }
}

static int prime_unified_encode_features(size_t rule_index,
                                         const size_t *accepted_indices,
                                         size_t accepted_count,
                                         const NNRuleCoverage *coverage,
                                         const uint64_t *covered,
                                         void *context,
                                         nn_real *features,
                                         size_t capacity)
{
    if (features == NULL || capacity < (size_t)UNIFIED_SCORER_FEATURE_COUNT) {
        return 0;
    }
    PrimeClosedLoopContext *prime = (PrimeClosedLoopContext *)context;
    if (prime == NULL || prime->rules == NULL || rule_index >= prime->rules->count) {
        return 0;
    }

    nn_real discovery_features[RULE_DISCOVERY_FEATURE_COUNT];
    if (!prime_closed_loop_encode_features(rule_index,
                                           accepted_indices,
                                           accepted_count,
                                           coverage,
                                           covered,
                                           context,
                                           discovery_features,
                                           (size_t)RULE_DISCOVERY_FEATURE_COUNT)) {
        return 0;
    }

    NNScorerEnvelopeRow spec = {
        .role = NN_SCORER_ROLE_RULE,
        .domain = NN_SCORER_DOMAIN_PRIME_RULE_PROPOSE,
        .candidate_index = rule_index,
        .candidate_count = prime->rules->count,
        .candidate_count_scale = 65536u,
        .include_role_tag = 1,
        .include_domain_tag = 1,
    };
    if (!nn_scorer_envelope_fill_base(features, capacity, &spec)) {
        return 0;
    }
    memcpy(&features[UNIFIED_F_RULE_FEATURES],
           discovery_features,
           (size_t)RULE_DISCOVERY_FEATURE_COUNT * sizeof(*features));
    return 1;
}

typedef enum {
    UNIFIED_ACTION_PRIME_GAP = 0,
    UNIFIED_ACTION_GRID_PATH = 1,
    UNIFIED_ACTION_DELAYED_CLAIM = 2,
    UNIFIED_ACTION_ARC_AGI = 3,
    UNIFIED_ACTION_TASK_COUNT = 4,
} UnifiedActionTask;

enum {
    UNIFIED_GRID_SIZE = 7,
    UNIFIED_GRID_CELLS = UNIFIED_GRID_SIZE * UNIFIED_GRID_SIZE,
    UNIFIED_GRID_ACTION_COUNT = 4,
    UNIFIED_GRID_INF = 1000000,
    UNIFIED_CLAIM_STATE_FEATURES = 6,
    UNIFIED_CLAIM_ACTION_COUNT = 4,
    UNIFIED_ARC_INPUT_WIDTH = 7,
    UNIFIED_ARC_INPUT_HEIGHT = 3,
    UNIFIED_ARC_INPUT_CELLS = UNIFIED_ARC_INPUT_WIDTH * UNIFIED_ARC_INPUT_HEIGHT,
    UNIFIED_ARC_OUTPUT_WIDTH = 3,
    UNIFIED_ARC_OUTPUT_HEIGHT = 3,
    UNIFIED_ARC_OUTPUT_CELLS = UNIFIED_ARC_OUTPUT_WIDTH * UNIFIED_ARC_OUTPUT_HEIGHT,
    UNIFIED_ARC_ACTION_COUNT = 10,
    UNIFIED_ARC_TRAIN_PAIR_COUNT = 3,
};

static const int UNIFIED_GRID_DX[UNIFIED_GRID_ACTION_COUNT] = {0, 1, 0, -1};
static const int UNIFIED_GRID_DY[UNIFIED_GRID_ACTION_COUNT] = {-1, 0, 1, 0};

static const uint8_t UNIFIED_ARC_0520FDE7_TRAIN_INPUTS[UNIFIED_ARC_TRAIN_PAIR_COUNT][UNIFIED_ARC_INPUT_CELLS] = {
    {1, 0, 0, 5, 0, 1, 0,
     0, 1, 0, 5, 1, 1, 1,
     1, 0, 0, 5, 0, 0, 0},
    {1, 1, 0, 5, 0, 1, 0,
     0, 0, 1, 5, 1, 1, 1,
     1, 1, 0, 5, 0, 1, 0},
    {0, 0, 1, 5, 0, 0, 0,
     1, 1, 0, 5, 1, 0, 1,
     0, 1, 1, 5, 1, 0, 1},
};

static const uint8_t UNIFIED_ARC_0520FDE7_TRAIN_OUTPUTS[UNIFIED_ARC_TRAIN_PAIR_COUNT][UNIFIED_ARC_OUTPUT_CELLS] = {
    {0, 0, 0,
     0, 2, 0,
     0, 0, 0},
    {0, 2, 0,
     0, 0, 2,
     0, 2, 0},
    {0, 0, 0,
     2, 0, 0,
     0, 0, 2},
};

typedef struct {
    uint8_t blocked[UNIFIED_GRID_CELLS];
    int sx;
    int sy;
    int gx;
    int gy;
} UnifiedGridState;

typedef struct {
    nn_real state[UNIFIED_CLAIM_STATE_FEATURES];
    size_t target;
    nn_real utility[UNIFIED_CLAIM_ACTION_COUNT];
} UnifiedClaimState;

typedef struct {
    size_t max_steps;
    size_t per_step_limit;
    size_t word_count;
    size_t step_count;
    size_t *accepted_counts;
    size_t *accepted_indices;
    uint64_t *covered_words;
    size_t *target_rule_indices;
    size_t *group_counts;
    size_t *group_indices;
} UnifiedRuleTrajectory;

static uint32_t unified_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static size_t unified_rng_bounded(uint32_t *state, size_t bound)
{
    if (bound == 0) {
        return 0;
    }
    return (size_t)(((uint64_t)unified_rng_next(state) * (uint64_t)bound) >> 32);
}

static void unified_fill_action_base(nn_real *row,
                                     UnifiedActionTask task,
                                     size_t action,
                                     size_t candidate_count)
{
    NNScorerDomain domain = NN_SCORER_DOMAIN_PRIME_GAP_ACTION;
    switch (task) {
    case UNIFIED_ACTION_PRIME_GAP:
        domain = NN_SCORER_DOMAIN_PRIME_GAP_ACTION;
        break;
    case UNIFIED_ACTION_GRID_PATH:
        domain = NN_SCORER_DOMAIN_GRID_PATH_ACTION;
        break;
    case UNIFIED_ACTION_DELAYED_CLAIM:
        domain = NN_SCORER_DOMAIN_DELAYED_CLAIM_ACTION;
        break;
    case UNIFIED_ACTION_ARC_AGI:
        domain = NN_SCORER_DOMAIN_ARC_AGI_ACTION;
        break;
    case UNIFIED_ACTION_TASK_COUNT:
        break;
    }
    NNScorerEnvelopeRow spec = {
        .role = NN_SCORER_ROLE_ACTION,
        .domain = domain,
        .candidate_index = action,
        .candidate_count = candidate_count,
        .candidate_count_scale = ACTION_COUNT,
        .include_role_tag = 1,
        .include_domain_tag = 1,
    };
    (void)nn_scorer_envelope_fill_base(row, (size_t)UNIFIED_SCORER_FEATURE_COUNT, &spec);
}

static size_t unified_fill_prime_action_sample(const GapDataset *dataset,
                                               double prime_log_scale,
                                               size_t index,
                                               nn_real *candidate_inputs,
                                               size_t *target_index)
{
    if (dataset == NULL || dataset->count == 0 || candidate_inputs == NULL ||
        target_index == NULL) {
        return 0;
    }
    size_t row_index = index % dataset->count;
    uint32_t p = dataset->prime[row_index];
    uint32_t target_gap = dataset->gap[row_index];
    *target_index = gap_to_action(target_gap);
    if (*target_index >= ACTION_COUNT) {
        return 0;
    }
    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        size_t candidate_gap = action_gap(action);
        uint64_t candidate = (uint64_t)p + (uint64_t)candidate_gap;
        nn_real *row = &candidate_inputs[action * UNIFIED_SCORER_FEATURE_COUNT];
        unified_fill_action_base(row, UNIFIED_ACTION_PRIME_GAP, action, ACTION_COUNT);
        row[UNIFIED_F_ACTION_COMMON + 0] =
            (nn_real)(log((double)p) / prime_log_scale);
        row[UNIFIED_F_ACTION_COMMON + 1] =
            (nn_real)((double)candidate_gap / (double)(G_MAX + 2));
        row[UNIFIED_F_ACTION_COMMON + 2] =
            action == OVERFLOW_ACTION ? (nn_real)1 : (nn_real)0;
        row[UNIFIED_F_ACTION_COMMON + 3] =
            (nn_real)(log1p((double)candidate_gap) / log1p((double)(G_MAX + 2)));
        row[UNIFIED_F_ACTION_COMMON + 4] =
            (nn_real)(log((double)candidate) / prime_log_scale);
        /* Match standalone prime-gap action features; otherwise the joint scorer
           sees only weak gap cues. */
        if (action == OVERFLOW_ACTION) {
            for (size_t i = 0; i < UNIFIED_PRIME_RESIDUE_FEATURES; ++i) {
                row[UNIFIED_F_ACTION_PAYLOAD + i] = (nn_real)0;
            }
            continue;
        }
        for (size_t i = 0; i < UNIFIED_PRIME_RESIDUE_FEATURES; ++i) {
            uint32_t mod = FEATURE_MODS[i];
            row[UNIFIED_F_ACTION_PAYLOAD + i] =
                candidate > mod && candidate % mod == 0 ? (nn_real)1 : (nn_real)0;
        }
    }
    return ACTION_COUNT;
}

static int unified_grid_cell_index(int x, int y)
{
    return y * UNIFIED_GRID_SIZE + x;
}

static int unified_grid_in_bounds(int x, int y)
{
    return x >= 0 && x < UNIFIED_GRID_SIZE && y >= 0 && y < UNIFIED_GRID_SIZE;
}

static size_t unified_grid_bfs_distances_from_goal(const UnifiedGridState *state, int *dist)
{
    int queue[UNIFIED_GRID_CELLS];
    size_t head = 0;
    size_t tail = 0;
    size_t expansions = 0;
    for (size_t i = 0; i < UNIFIED_GRID_CELLS; ++i) {
        dist[i] = UNIFIED_GRID_INF;
    }
    int goal = unified_grid_cell_index(state->gx, state->gy);
    dist[goal] = 0;
    queue[tail++] = goal;
    while (head < tail) {
        int cell = queue[head++];
        int x = cell % UNIFIED_GRID_SIZE;
        int y = cell / UNIFIED_GRID_SIZE;
        ++expansions;
        for (size_t action = 0; action < UNIFIED_GRID_ACTION_COUNT; ++action) {
            int nx = x + UNIFIED_GRID_DX[action];
            int ny = y + UNIFIED_GRID_DY[action];
            if (!unified_grid_in_bounds(nx, ny)) {
                continue;
            }
            int next = unified_grid_cell_index(nx, ny);
            if (state->blocked[next] || dist[next] != UNIFIED_GRID_INF) {
                continue;
            }
            dist[next] = dist[cell] + 1;
            queue[tail++] = next;
        }
    }
    return expansions;
}

static int unified_generate_grid_state(uint64_t seed, UnifiedGridState *state)
{
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ UINT64_C(0x6d2b79f5));
    for (size_t attempt = 0; attempt < 64; ++attempt) {
        memset(state, 0, sizeof(*state));
        state->sx = (int)(unified_rng_next(&rng) % UNIFIED_GRID_SIZE);
        state->sy = (int)(unified_rng_next(&rng) % UNIFIED_GRID_SIZE);
        do {
            state->gx = (int)(unified_rng_next(&rng) % UNIFIED_GRID_SIZE);
            state->gy = (int)(unified_rng_next(&rng) % UNIFIED_GRID_SIZE);
        } while (state->gx == state->sx && state->gy == state->sy);
        for (int y = 0; y < UNIFIED_GRID_SIZE; ++y) {
            for (int x = 0; x < UNIFIED_GRID_SIZE; ++x) {
                if ((x == state->sx && y == state->sy) ||
                    (x == state->gx && y == state->gy)) {
                    continue;
                }
                state->blocked[unified_grid_cell_index(x, y)] =
                    (unified_rng_next(&rng) % 100u) < 23u ? 1u : 0u;
            }
        }
        int dist[UNIFIED_GRID_CELLS];
        unified_grid_bfs_distances_from_goal(state, dist);
        int start = unified_grid_cell_index(state->sx, state->sy);
        if (dist[start] > 0 && dist[start] < UNIFIED_GRID_INF) {
            return 1;
        }
    }
    return 0;
}

static size_t unified_fill_grid_action_sample(uint64_t seed,
                                              nn_real *candidate_inputs,
                                              size_t *target_index)
{
    UnifiedGridState state;
    if (!unified_generate_grid_state(seed, &state)) {
        return 0;
    }
    int dist[UNIFIED_GRID_CELLS];
    unified_grid_bfs_distances_from_goal(&state, dist);
    int best_dist = UNIFIED_GRID_INF;
    int manhattan_before = abs(state.gx - state.sx) + abs(state.gy - state.sy);
    *target_index = UNIFIED_GRID_ACTION_COUNT;
    for (size_t action = 0; action < UNIFIED_GRID_ACTION_COUNT; ++action) {
        int nx = state.sx + UNIFIED_GRID_DX[action];
        int ny = state.sy + UNIFIED_GRID_DY[action];
        int out_of_bounds = !unified_grid_in_bounds(nx, ny);
        int blocked_next = 0;
        int next_dist = UNIFIED_GRID_INF;
        int feature_x = state.sx;
        int feature_y = state.sy;
        if (!out_of_bounds) {
            int next_cell = unified_grid_cell_index(nx, ny);
            blocked_next = state.blocked[next_cell] ? 1 : 0;
            feature_x = nx;
            feature_y = ny;
            if (!blocked_next) {
                next_dist = dist[next_cell];
            }
        }
        if (next_dist < UNIFIED_GRID_INF &&
            (next_dist < best_dist ||
             (next_dist == best_dist && *target_index == UNIFIED_GRID_ACTION_COUNT))) {
            best_dist = next_dist;
            *target_index = action;
        }

        int manhattan_after = abs(state.gx - feature_x) + abs(state.gy - feature_y);
        nn_real *row = &candidate_inputs[action * UNIFIED_SCORER_FEATURE_COUNT];
        unified_fill_action_base(row, UNIFIED_ACTION_GRID_PATH, action, UNIFIED_GRID_ACTION_COUNT);
        row[UNIFIED_F_ACTION_COMMON + 0] = (nn_real)((double)state.sx / (double)(UNIFIED_GRID_SIZE - 1));
        row[UNIFIED_F_ACTION_COMMON + 1] = (nn_real)((double)state.sy / (double)(UNIFIED_GRID_SIZE - 1));
        row[UNIFIED_F_ACTION_COMMON + 2] = (nn_real)((double)state.gx / (double)(UNIFIED_GRID_SIZE - 1));
        row[UNIFIED_F_ACTION_COMMON + 3] = (nn_real)((double)state.gy / (double)(UNIFIED_GRID_SIZE - 1));
        row[UNIFIED_F_ACTION_COMMON + 4] =
            (nn_real)((double)(state.gx - feature_x) / (double)(UNIFIED_GRID_SIZE - 1));
        row[UNIFIED_F_ACTION_COMMON + 5] =
            (nn_real)((double)(state.gy - feature_y) / (double)(UNIFIED_GRID_SIZE - 1));
        row[UNIFIED_F_ACTION_COMMON + 6] =
            (nn_real)((double)manhattan_after / (double)(2 * (UNIFIED_GRID_SIZE - 1)));
        row[UNIFIED_F_ACTION_COMMON + 7] =
            (nn_real)((double)(manhattan_before - manhattan_after) / 2.0);
        row[UNIFIED_F_ACTION_STATE + 0] = (nn_real)UNIFIED_GRID_DX[action];
        row[UNIFIED_F_ACTION_STATE + 1] = (nn_real)UNIFIED_GRID_DY[action];
        row[UNIFIED_F_ACTION_STATE + 2] = out_of_bounds ? (nn_real)1 : (nn_real)0;
        row[UNIFIED_F_ACTION_STATE + 3] = blocked_next ? (nn_real)1 : (nn_real)0;
        for (size_t cell = 0; cell < UNIFIED_GRID_CELLS; ++cell) {
            row[UNIFIED_F_RULE_FEATURES + cell] = state.blocked[cell] ? (nn_real)1 : (nn_real)0;
        }
        for (size_t i = 0; i < UNIFIED_GRID_ACTION_COUNT; ++i) {
            row[UNIFIED_F_RULE_FEATURES + UNIFIED_GRID_CELLS + i] =
                action == i ? (nn_real)1 : (nn_real)0;
        }
    }
    return *target_index < UNIFIED_GRID_ACTION_COUNT ? UNIFIED_GRID_ACTION_COUNT : 0;
}

static UnifiedClaimState unified_generate_claim_state(uint64_t seed)
{
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ UINT64_C(0xa511e9b3));
    UnifiedClaimState state;
    for (size_t i = 0; i < UNIFIED_CLAIM_STATE_FEATURES; ++i) {
        state.state[i] = (nn_real)((double)(unified_rng_next(&rng) % 10000u) / 9999.0);
    }
    nn_real reliability = state.state[0];
    nn_real evidence = state.state[1];
    nn_real contradiction = state.state[2];
    nn_real age = state.state[3];
    nn_real history = state.state[4];
    nn_real noise = state.state[5];
    state.utility[0] = reliability + (nn_real)0.8 * evidence + (nn_real)0.4 * history -
                       (nn_real)1.1 * contradiction - (nn_real)0.2 * age;
    state.utility[1] = (nn_real)0.35 + contradiction + (nn_real)0.25 * age -
                       (nn_real)0.5 * evidence;
    state.utility[2] = (nn_real)0.25 + (nn_real)0.7 * noise + (nn_real)0.5 * contradiction -
                       (nn_real)0.3 * history;
    state.utility[3] = (nn_real)0.45 + (nn_real)0.5 * reliability + (nn_real)0.3 * evidence -
                       (nn_real)0.4 * contradiction;
    state.target = 0;
    for (size_t action = 1; action < UNIFIED_CLAIM_ACTION_COUNT; ++action) {
        if (state.utility[action] > state.utility[state.target]) {
            state.target = action;
        }
    }
    return state;
}

static size_t unified_fill_claim_action_sample(uint64_t seed,
                                               nn_real *candidate_inputs,
                                               size_t *target_index)
{
    UnifiedClaimState state = unified_generate_claim_state(seed);
    *target_index = state.target;
    for (size_t action = 0; action < UNIFIED_CLAIM_ACTION_COUNT; ++action) {
        nn_real *row = &candidate_inputs[action * UNIFIED_SCORER_FEATURE_COUNT];
        unified_fill_action_base(row, UNIFIED_ACTION_DELAYED_CLAIM, action, UNIFIED_CLAIM_ACTION_COUNT);
        for (size_t i = 0; i < UNIFIED_CLAIM_STATE_FEATURES; ++i) {
            row[UNIFIED_F_ACTION_COMMON + i] = state.state[i];
        }
        for (size_t i = 0; i < UNIFIED_CLAIM_ACTION_COUNT; ++i) {
            row[UNIFIED_F_RULE_FEATURES + i] = action == i ? (nn_real)1 : (nn_real)0;
        }
    }
    return UNIFIED_CLAIM_ACTION_COUNT;
}

static size_t unified_arc_select_balanced_cell(const uint8_t *output, uint64_t sample)
{
    int want_foreground = (sample & UINT64_C(1)) == 0;
    size_t match_count = 0;
    for (size_t cell = 0; cell < UNIFIED_ARC_OUTPUT_CELLS; ++cell) {
        int foreground = output[cell] != 0;
        if (foreground == want_foreground) {
            ++match_count;
        }
    }
    if (match_count == 0) {
        return (size_t)(sample % UNIFIED_ARC_OUTPUT_CELLS);
    }
    size_t wanted = (size_t)((sample >> 1) % match_count);
    for (size_t cell = 0; cell < UNIFIED_ARC_OUTPUT_CELLS; ++cell) {
        int foreground = output[cell] != 0;
        if (foreground != want_foreground) {
            continue;
        }
        if (wanted == 0) {
            return cell;
        }
        --wanted;
    }
    return 0;
}

static size_t unified_fill_arc_action_sample(uint64_t sample,
                                             nn_real *candidate_inputs,
                                             size_t *target_index)
{
    size_t pair = (size_t)(sample % UNIFIED_ARC_TRAIN_PAIR_COUNT);
    const uint8_t *input = UNIFIED_ARC_0520FDE7_TRAIN_INPUTS[pair];
    const uint8_t *output = UNIFIED_ARC_0520FDE7_TRAIN_OUTPUTS[pair];
    size_t cell = unified_arc_select_balanced_cell(output, sample / UNIFIED_ARC_TRAIN_PAIR_COUNT);
    size_t output_x = cell % UNIFIED_ARC_OUTPUT_WIDTH;
    size_t output_y = cell / UNIFIED_ARC_OUTPUT_WIDTH;
    *target_index = (size_t)output[cell];
    for (size_t action = 0; action < UNIFIED_ARC_ACTION_COUNT; ++action) {
        nn_real *row = &candidate_inputs[action * UNIFIED_SCORER_FEATURE_COUNT];
        unified_fill_action_base(row, UNIFIED_ACTION_ARC_AGI, action, UNIFIED_ARC_ACTION_COUNT);
        row[UNIFIED_F_ACTION_COMMON + 0] =
            (nn_real)((double)output_x / (double)(UNIFIED_ARC_OUTPUT_WIDTH - 1));
        row[UNIFIED_F_ACTION_COMMON + 1] =
            (nn_real)((double)output_y / (double)(UNIFIED_ARC_OUTPUT_HEIGHT - 1));
        row[UNIFIED_F_ACTION_COMMON + 2] = (nn_real)((double)UNIFIED_ARC_INPUT_WIDTH / 30.0);
        row[UNIFIED_F_ACTION_COMMON + 3] = (nn_real)((double)UNIFIED_ARC_INPUT_HEIGHT / 30.0);
        row[UNIFIED_F_ACTION_COMMON + 4] = (nn_real)((double)UNIFIED_ARC_OUTPUT_WIDTH / 30.0);
        row[UNIFIED_F_ACTION_COMMON + 5] = (nn_real)((double)UNIFIED_ARC_OUTPUT_HEIGHT / 30.0);
        row[UNIFIED_F_ACTION_COMMON + 6] =
            (nn_real)((double)cell / (double)(UNIFIED_ARC_OUTPUT_CELLS - 1));
        row[UNIFIED_F_ACTION_COMMON + 7] = action == 0 ? (nn_real)1 : (nn_real)0;
        row[UNIFIED_F_ACTION_COMMON + 8] = (nn_real)((double)action / 9.0);
        for (size_t i = 0; i < UNIFIED_ARC_INPUT_CELLS; ++i) {
            row[UNIFIED_F_RULE_FEATURES + i] = (nn_real)((double)input[i] / 9.0);
        }
        row[UNIFIED_F_RULE_FEATURES + UNIFIED_ARC_INPUT_CELLS +
            output_y * UNIFIED_ARC_OUTPUT_WIDTH + output_x] = (nn_real)1;
        row[UNIFIED_F_RULE_FEATURES + UNIFIED_ARC_INPUT_CELLS +
            UNIFIED_ARC_OUTPUT_CELLS + action] = (nn_real)1;
    }
    return UNIFIED_ARC_ACTION_COUNT;
}

static size_t unified_fill_action_sample(const GapDataset *prime_train,
                                         double prime_log_scale,
                                         UnifiedActionTask task,
                                         uint64_t sample,
                                         nn_real *candidate_inputs,
                                         size_t *target_index)
{
    switch (task) {
    case UNIFIED_ACTION_PRIME_GAP:
        return unified_fill_prime_action_sample(prime_train,
                                                prime_log_scale,
                                                (size_t)sample,
                                                candidate_inputs,
                                                target_index);
    case UNIFIED_ACTION_GRID_PATH:
        return unified_fill_grid_action_sample(7000u + sample, candidate_inputs, target_index);
    case UNIFIED_ACTION_DELAYED_CLAIM:
        return unified_fill_claim_action_sample(1000u + sample, candidate_inputs, target_index);
    case UNIFIED_ACTION_ARC_AGI:
        return unified_fill_arc_action_sample(5000u + sample, candidate_inputs, target_index);
    case UNIFIED_ACTION_TASK_COUNT:
        break;
    }
    return 0;
}

static void unified_rule_trajectory_free(UnifiedRuleTrajectory *trajectory)
{
    if (trajectory == NULL) {
        return;
    }
    free(trajectory->accepted_counts);
    free(trajectory->accepted_indices);
    free(trajectory->covered_words);
    free(trajectory->target_rule_indices);
    free(trajectory->group_counts);
    free(trajectory->group_indices);
    memset(trajectory, 0, sizeof(*trajectory));
}

static int unified_build_rule_trajectory(const NNRuleCoverage *coverage,
                                         const RuleSearchConfig *config,
                                         PrimeClosedLoopContext *context,
                                         size_t eligible_count,
                                         UnifiedRuleTrajectory *trajectory)
{
    memset(trajectory, 0, sizeof(*trajectory));
    if (coverage == NULL || config == NULL || context == NULL ||
        eligible_count == 0 || coverage->word_count == 0) {
        return 0;
    }
    size_t max_steps = config->max_rules < eligible_count ? config->max_rules : eligible_count;
    size_t per_step_limit =
        eligible_count < (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP
            ? eligible_count
            : (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP;
    if (max_steps == 0 || per_step_limit == 0) {
        return 0;
    }

    trajectory->max_steps = max_steps;
    trajectory->per_step_limit = per_step_limit;
    trajectory->word_count = coverage->word_count;
    trajectory->accepted_counts = calloc(max_steps, sizeof(*trajectory->accepted_counts));
    trajectory->accepted_indices = calloc(max_steps * max_steps, sizeof(*trajectory->accepted_indices));
    trajectory->covered_words = calloc(max_steps * coverage->word_count, sizeof(*trajectory->covered_words));
    trajectory->target_rule_indices = calloc(max_steps, sizeof(*trajectory->target_rule_indices));
    trajectory->group_counts = calloc(max_steps, sizeof(*trajectory->group_counts));
    trajectory->group_indices = calloc(max_steps * per_step_limit, sizeof(*trajectory->group_indices));
    uint8_t *selected = calloc(coverage->rule_count, sizeof(*selected));
    uint64_t *covered = calloc(coverage->word_count, sizeof(*covered));
    size_t *gains = calloc(coverage->rule_count, sizeof(*gains));
    size_t *accepted_indices = calloc(max_steps, sizeof(*accepted_indices));
    if (trajectory->accepted_counts == NULL || trajectory->accepted_indices == NULL ||
        trajectory->covered_words == NULL || trajectory->target_rule_indices == NULL ||
        trajectory->group_counts == NULL || trajectory->group_indices == NULL ||
        selected == NULL || covered == NULL || gains == NULL || accepted_indices == NULL) {
        free(selected);
        free(covered);
        free(gains);
        free(accepted_indices);
        unified_rule_trajectory_free(trajectory);
        return 0;
    }

    size_t accepted_count = 0;
    for (size_t step = 0; step < max_steps; ++step) {
        size_t best_index = (size_t)-1;
        size_t best_gain = 0;
        memset(gains, 0, coverage->rule_count * sizeof(*gains));
        for (size_t i = 0; i < coverage->rule_count; ++i) {
            if (selected[i] ||
                !prime_closed_loop_candidate_allowed(i, accepted_indices, accepted_count, context)) {
                continue;
            }
            size_t gain = nn_rule_coverage_gain(coverage, i, covered);
            gains[i] = gain;
            if (gain > best_gain) {
                best_gain = gain;
                best_index = i;
            }
        }
        if (best_index == (size_t)-1 || best_gain == 0) {
            break;
        }

        trajectory->accepted_counts[step] = accepted_count;
        memcpy(&trajectory->accepted_indices[step * max_steps],
               accepted_indices,
               accepted_count * sizeof(*accepted_indices));
        memcpy(&trajectory->covered_words[step * coverage->word_count],
               covered,
               coverage->word_count * sizeof(*covered));
        trajectory->target_rule_indices[step] = best_index;

        size_t kept = 0;
        trajectory->group_indices[step * per_step_limit + kept++] = best_index;
        for (size_t i = 0; i < coverage->rule_count && kept < per_step_limit; ++i) {
            if (i == best_index || gains[i] == 0 || selected[i] ||
                !prime_closed_loop_candidate_allowed(i, accepted_indices, accepted_count, context)) {
                continue;
            }
            trajectory->group_indices[step * per_step_limit + kept++] = i;
        }
        for (size_t i = 0; i < coverage->rule_count && kept < per_step_limit; ++i) {
            if (i == best_index || gains[i] != 0 || selected[i] ||
                !prime_closed_loop_candidate_allowed(i, accepted_indices, accepted_count, context)) {
                continue;
            }
            uint64_t hash = mix_rule_hash_u64((uint64_t)i,
                                              ((uint64_t)step << 32) ^
                                                  (uint64_t)config->proposer_seed);
            if ((hash % 17u) != 0u && kept + 1u < per_step_limit) {
                continue;
            }
            trajectory->group_indices[step * per_step_limit + kept++] = i;
        }
        trajectory->group_counts[step] = kept;
        ++trajectory->step_count;

        selected[best_index] = 1;
        size_t write = 0;
        for (size_t i = 0; i < accepted_count; ++i) {
            if (prime_closed_loop_keep_accepted(accepted_indices[i], best_index, context)) {
                accepted_indices[write++] = accepted_indices[i];
            }
        }
        accepted_count = write;
        if (accepted_count < max_steps) {
            accepted_indices[accepted_count++] = best_index;
        }
        nn_rule_coverage_add_rule_to_residual(coverage, best_index, covered);
    }

    free(selected);
    free(covered);
    free(gains);
    free(accepted_indices);
    if (trajectory->step_count == 0) {
        unified_rule_trajectory_free(trajectory);
        return 0;
    }
    return 1;
}

static int unified_train_rule_softmax_step(NeuralNet *net,
                                           const NNRuleCoverage *coverage,
                                           PrimeClosedLoopContext *context,
                                           const UnifiedRuleTrajectory *trajectory,
                                           size_t step,
                                           nn_real *candidate_inputs,
                                           nn_real learning_rate,
                                           nn_real *loss_out,
                                           size_t *row_count_out)
{
    if (net == NULL || coverage == NULL || context == NULL || trajectory == NULL ||
        candidate_inputs == NULL || step >= trajectory->step_count) {
        return 0;
    }
    size_t group_count = trajectory->group_counts[step];
    if (group_count == 0) {
        return 0;
    }
    const size_t *accepted_indices =
        &trajectory->accepted_indices[step * trajectory->max_steps];
    size_t accepted_count = trajectory->accepted_counts[step];
    const uint64_t *covered = &trajectory->covered_words[step * trajectory->word_count];
    for (size_t group = 0; group < group_count; ++group) {
        size_t rule_index = trajectory->group_indices[step * trajectory->per_step_limit + group];
        if (!prime_unified_encode_features(rule_index,
                                           accepted_indices,
                                           accepted_count,
                                           coverage,
                                           covered,
                                           context,
                                           &candidate_inputs[group * UNIFIED_SCORER_FEATURE_COUNT],
                                           (size_t)UNIFIED_SCORER_FEATURE_COUNT)) {
            return 0;
        }
    }
    nn_real loss = nn_train_action_scorer(net,
                                          candidate_inputs,
                                          group_count,
                                          0,
                                          learning_rate);
    if (isnan((double)loss)) {
        return 0;
    }
    if (loss_out != NULL) {
        *loss_out = loss;
    }
    if (row_count_out != NULL) {
        *row_count_out = group_count;
    }
    return 1;
}

static NeuralNet *unified_create_shared_scorer(const RuleSearchConfig *config)
{
    size_t layers[4];
    layers[0] = (size_t)UNIFIED_SCORER_FEATURE_COUNT;
    layers[1] = rule_search_effective_unified_hidden1(config);
    layers[2] = rule_search_effective_unified_hidden2(config);
    layers[3] = 1u;
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = 4u;
    nn_config.seed = config->proposer_seed ^ 520637u;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    return nn_create_config(&nn_config);
}

static int unified_train_shared_action_rule_scorer(NeuralNet *net,
                                                   const GapDataset *prime_train,
                                                   const RuleSearchConfig *config,
                                                   const NNRuleCoverage *coverage,
                                                   PrimeClosedLoopContext *context,
                                                   const UnifiedRuleTrajectory *trajectory,
                                                   RuleSearchMetrics *metrics)
{
    size_t rule_updates = config->max_rules * (size_t)UNIFIED_TRANSFER_RULE_UPDATES_PER_BUDGET;
    if (rule_updates < (size_t)UNIFIED_TRANSFER_MIN_RULE_UPDATES) {
        rule_updates = (size_t)UNIFIED_TRANSFER_MIN_RULE_UPDATES;
    }
    size_t action_updates = rule_updates;
    nn_real *action_inputs =
        calloc((size_t)ACTION_COUNT * (size_t)UNIFIED_SCORER_FEATURE_COUNT, sizeof(*action_inputs));
    nn_real *rule_inputs =
        calloc(trajectory->per_step_limit * (size_t)UNIFIED_SCORER_FEATURE_COUNT, sizeof(*rule_inputs));
    if (action_inputs == NULL || rule_inputs == NULL) {
        free(action_inputs);
        free(rule_inputs);
        return 0;
    }

    uint32_t rng = (uint32_t)(config->proposer_seed ^ UINT32_C(0x31415926));
    double prime_log_scale = log((double)config->test_limit + (double)G_MAX + 2.0);
    double action_loss_sum = 0.0;
    double rule_loss_sum = 0.0;
    size_t action_done = 0;
    size_t rule_done = 0;
    size_t rule_rows = 0;
    size_t total_updates = action_updates + rule_updates;
    for (size_t update = 0; update < total_updates; ++update) {
        int train_rule = (update % 2u) == 0;
        if (train_rule && rule_done < rule_updates) {
            size_t step = unified_rng_bounded(&rng, trajectory->step_count);
            nn_real loss = 0;
            size_t rows = 0;
            if (!unified_train_rule_softmax_step(net,
                                                 coverage,
                                                 context,
                                                 trajectory,
                                                 step,
                                                 rule_inputs,
                                                 (nn_real)0.006,
                                                 &loss,
                                                 &rows)) {
                free(action_inputs);
                free(rule_inputs);
                return 0;
            }
            rule_loss_sum += (double)loss;
            rule_rows += rows;
            ++rule_done;
            continue;
        }

        if (action_done < action_updates) {
            UnifiedActionTask task =
                (UnifiedActionTask)unified_rng_bounded(&rng, (size_t)UNIFIED_ACTION_TASK_COUNT);
            uint64_t sample = (uint64_t)unified_rng_next(&rng);
            if (task == UNIFIED_ACTION_PRIME_GAP && prime_train->count != 0) {
                sample %= (uint64_t)prime_train->count;
            }
            size_t target_index = 0;
            size_t candidate_count = unified_fill_action_sample(prime_train,
                                                                prime_log_scale,
                                                                task,
                                                                sample,
                                                                action_inputs,
                                                                &target_index);
            if (candidate_count == 0 || target_index >= candidate_count) {
                free(action_inputs);
                free(rule_inputs);
                return 0;
            }
            nn_real loss = nn_train_action_scorer(net,
                                                  action_inputs,
                                                  candidate_count,
                                                  target_index,
                                                  (nn_real)0.006);
            if (isnan((double)loss)) {
                free(action_inputs);
                free(rule_inputs);
                return 0;
            }
            action_loss_sum += (double)loss;
            ++action_done;
        }
    }

    metrics->unified_action_updates = action_done;
    metrics->unified_rule_updates = rule_done;
    metrics->residual_train_samples = rule_rows;
    metrics->proposer_loss = rule_done == 0 ? 0.0 : rule_loss_sum / (double)rule_done;
    printf("unified_transfer action_updates=%zu rule_updates=%zu action_loss=%.4f rule_loss=%.4f\n",
           action_done,
           rule_done,
           action_done == 0 ? 0.0 : action_loss_sum / (double)action_done,
           metrics->proposer_loss);

    free(action_inputs);
    free(rule_inputs);
    return action_done == action_updates && rule_done == rule_updates;
}

PRIME_GAP_TEST_API int select_expanded_rules_closed_loop(ExpandedRuleSet *rules,
                                                         const RuleSearchConfig *config,
                                                         const RuleSearchGrammar *grammar,
                                                         const GapDataset *dataset,
                                                         size_t max_states,
                                                         ExpandedRuleSet *accepted,
                                                         RuleSearchMetrics *metrics)
{
    memset(accepted, 0, sizeof(*accepted));
    accepted->items = calloc(config->max_rules, sizeof(*accepted->items));
    if (accepted->items == NULL) {
        return 0;
    }

    summarize_expanded_rule_candidates(rules, metrics);

    NNRuleCoverage coverage;
    size_t samples = 0;
    if (!build_rule_coverage(config, grammar, rules, dataset, max_states, &coverage, &samples)) {
        return 0;
    }

    PrimeClosedLoopContext context;
    context.rules = rules;
    context.config = config;
    context.max_modulus = max_rule_search_modulus(config);

    size_t eligible_count = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        if (prime_closed_loop_candidate_allowed(i, NULL, 0, &context)) {
            ++eligible_count;
        }
    }
    metrics->softmax_group_size =
        eligible_count < (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP
            ? eligible_count
            : (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP;

    NNClosedLoopRulePolicyConfig policy_config;
    nn_closed_loop_rule_policy_config_defaults(&policy_config,
                                               (size_t)RULE_DISCOVERY_FEATURE_COUNT);
    policy_config.max_rules = config->max_rules;
    policy_config.max_train_rules_per_step = (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP;
    policy_config.epochs = (size_t)RULE_SEARCH_SCORER_EPOCHS;
    policy_config.hidden1 = rule_search_effective_hidden1(config);
    policy_config.hidden2 = rule_search_effective_hidden2(config);
    policy_config.seed = config->proposer_seed ^ 119417u;
    policy_config.verbose = 1;
    policy_config.feature_schema = PRIME_CLOSED_LOOP_RULE_FEATURES;
    policy_config.feature_schema_count = sizeof(PRIME_CLOSED_LOOP_RULE_FEATURES) /
                                         sizeof(PRIME_CLOSED_LOOP_RULE_FEATURES[0]);
    policy_config.leakage_policy = NN_LEAKAGE_GATE_ALLOWED;

    NNClosedLoopRulePolicy policy;
    if (!nn_closed_loop_rule_policy_train(&coverage,
                                          &policy_config,
                                          prime_closed_loop_encode_features,
                                          prime_closed_loop_candidate_allowed,
                                          prime_closed_loop_keep_accepted,
                                          &context,
                                          &policy)) {
        nn_rule_coverage_free(&coverage);
        return 0;
    }
    metrics->proposer_loss = policy.loss;
    metrics->residual_train_samples = policy.sample_count;
    metrics->parameters = nn_parameter_count(policy.net);
    metrics->forward_flops = nn_forward_flop_count(policy.net);

    nn_real *scores = calloc(rules->count, sizeof(*scores));
    size_t *accepted_indices = calloc(config->max_rules, sizeof(*accepted_indices));
    size_t accepted_count = 0;
    size_t proposal_calls = 0;
    if (scores == NULL || accepted_indices == NULL) {
        free(scores);
        free(accepted_indices);
        nn_closed_loop_rule_policy_free(&policy);
        nn_rule_coverage_free(&coverage);
        return 0;
    }

    int ok = nn_closed_loop_rule_policy_select(&coverage,
                                               &policy_config,
                                               prime_closed_loop_encode_features,
                                               prime_closed_loop_candidate_allowed,
                                               prime_closed_loop_keep_accepted,
                                               &context,
                                               &policy,
                                               scores,
                                               accepted_indices,
                                               config->max_rules,
                                               &accepted_count,
                                               &proposal_calls);
    if (!ok) {
        free(scores);
        free(accepted_indices);
        nn_closed_loop_rule_policy_free(&policy);
        nn_rule_coverage_free(&coverage);
        return 0;
    }
    metrics->proposer_calls = proposal_calls;
    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].model_score = scores[i];
    }
    for (size_t i = 0; i < accepted_count; ++i) {
        size_t rule_index = accepted_indices[i];
        if (rule_index < rules->count && accepted->count < config->max_rules) {
            rules->items[rule_index].accepted = 1;
            accepted->items[accepted->count++] = rules->items[rule_index];
        }
    }

    mark_expanded_rule_acceptance(rules, accepted, metrics);

    free(scores);
    free(accepted_indices);
    nn_closed_loop_rule_policy_free(&policy);
    nn_rule_coverage_free(&coverage);
    return 1;
}

PRIME_GAP_TEST_API int select_expanded_rules_unified(ExpandedRuleSet *rules,
                                                     const RuleSearchConfig *config,
                                                     const RuleSearchGrammar *grammar,
                                                     const GapDataset *dataset,
                                                     size_t max_states,
                                                     ExpandedRuleSet *accepted,
                                                     RuleSearchMetrics *metrics)
{
    memset(accepted, 0, sizeof(*accepted));
    accepted->items = calloc(config->max_rules, sizeof(*accepted->items));
    if (accepted->items == NULL) {
        return 0;
    }

    summarize_expanded_rule_candidates(rules, metrics);

    NNRuleCoverage coverage;
    size_t samples = 0;
    if (!build_rule_coverage(config, grammar, rules, dataset, max_states, &coverage, &samples)) {
        return 0;
    }

    PrimeClosedLoopContext context;
    context.rules = rules;
    context.config = config;
    context.max_modulus = max_rule_search_modulus(config);

    size_t eligible_count = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        if (prime_closed_loop_candidate_allowed(i, NULL, 0, &context)) {
            ++eligible_count;
        }
    }
    metrics->softmax_group_size =
        eligible_count < (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP
            ? eligible_count
            : (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP;

    NNFeatureSchema feature_schema[UNIFIED_SCORER_FEATURE_COUNT];
    char feature_names[UNIFIED_SCORER_FEATURE_COUNT][64];
    prime_unified_feature_schema(feature_schema, feature_names);

    NNClosedLoopRulePolicyConfig policy_config;
    nn_closed_loop_rule_policy_config_defaults(&policy_config,
                                               (size_t)UNIFIED_SCORER_FEATURE_COUNT);
    policy_config.max_rules = config->max_rules;
    policy_config.max_train_rules_per_step = (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP;
    policy_config.epochs = (size_t)RULE_SEARCH_SCORER_EPOCHS;
    policy_config.hidden1 = rule_search_effective_unified_hidden1(config);
    policy_config.hidden2 = rule_search_effective_unified_hidden2(config);
    policy_config.seed = config->proposer_seed ^ 520637u;
    policy_config.verbose = 1;
    policy_config.feature_schema = feature_schema;
    policy_config.feature_schema_count = (size_t)UNIFIED_SCORER_FEATURE_COUNT;
    policy_config.leakage_policy = NN_LEAKAGE_GATE_ALLOWED;

    UnifiedRuleTrajectory trajectory;
    if (!unified_build_rule_trajectory(&coverage,
                                       config,
                                       &context,
                                       eligible_count,
                                       &trajectory)) {
        nn_rule_coverage_free(&coverage);
        return 0;
    }

    NeuralNet *net = unified_create_shared_scorer(config);
    if (net == NULL) {
        unified_rule_trajectory_free(&trajectory);
        nn_rule_coverage_free(&coverage);
        return 0;
    }
    if (!unified_train_shared_action_rule_scorer(net,
                                                 dataset,
                                                 config,
                                                 &coverage,
                                                 &context,
                                                 &trajectory,
                                                 metrics)) {
        nn_free(net);
        unified_rule_trajectory_free(&trajectory);
        nn_rule_coverage_free(&coverage);
        return 0;
    }
    metrics->parameters = nn_parameter_count(net);
    metrics->forward_flops = nn_forward_flop_count(net);

    NNClosedLoopRulePolicy policy;
    nn_closed_loop_rule_policy_clear(&policy);
    policy.net = net;
    policy.loss = metrics->proposer_loss;
    policy.sample_count = metrics->residual_train_samples;

    nn_real *scores = calloc(rules->count, sizeof(*scores));
    size_t *accepted_indices = calloc(config->max_rules, sizeof(*accepted_indices));
    size_t accepted_count = 0;
    size_t proposal_calls = 0;
    if (scores == NULL || accepted_indices == NULL) {
        free(scores);
        free(accepted_indices);
        nn_free(net);
        unified_rule_trajectory_free(&trajectory);
        nn_rule_coverage_free(&coverage);
        return 0;
    }

    int ok = nn_closed_loop_rule_policy_select(&coverage,
                                               &policy_config,
                                               prime_unified_encode_features,
                                               prime_closed_loop_candidate_allowed,
                                               prime_closed_loop_keep_accepted,
                                               &context,
                                               &policy,
                                               scores,
                                               accepted_indices,
                                               config->max_rules,
                                               &accepted_count,
                                               &proposal_calls);
    if (!ok) {
        free(scores);
        free(accepted_indices);
        nn_free(net);
        unified_rule_trajectory_free(&trajectory);
        nn_rule_coverage_free(&coverage);
        return 0;
    }
    metrics->proposer_calls = proposal_calls;
    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].model_score = scores[i];
    }
    for (size_t i = 0; i < accepted_count; ++i) {
        size_t rule_index = accepted_indices[i];
        if (rule_index < rules->count && accepted->count < config->max_rules) {
            rules->items[rule_index].accepted = 1;
            accepted->items[accepted->count++] = rules->items[rule_index];
        }
    }

    mark_expanded_rule_acceptance(rules, accepted, metrics);

    free(scores);
    free(accepted_indices);
    nn_free(net);
    unified_rule_trajectory_free(&trajectory);
    nn_rule_coverage_free(&coverage);
    return 1;
}

static int select_expanded_rules(ExpandedRuleSet *rules,
                                 size_t max_rules,
                                 ExpandedRuleSet *accepted,
                                 RuleSearchMetrics *metrics)
{
    memset(accepted, 0, sizeof(*accepted));
    accepted->items = calloc(max_rules, sizeof(*accepted->items));
    if (accepted->items == NULL) {
        return 0;
    }

    qsort(rules->items, rules->count, sizeof(*rules->items), compare_expanded_rules_model);

    metrics->proposed_rules = rules->count;
    metrics->proposer_calls = rules->count;
    for (size_t i = 0; i < rules->count; ++i) {
        ExpandedRule *rule = &rules->items[i];
        if (rule->train_false_rejections != 0 || rule->validation_false_rejections != 0) {
            ++metrics->unsafe_rules;
            continue;
        }
        if (rule->train_rejects == 0 && rule->validation_rejects == 0) {
            ++metrics->no_support_rules;
            continue;
        }
        ++metrics->verified_rules;
        if (!expanded_rule_has_deterministic_certificate(rule)) {
            ++metrics->uncertified_rules;
            continue;
        }
        if (expanded_rule_is_covered(accepted, rule)) {
            continue;
        }
        expanded_rule_remove_subsets(accepted, rule);
        if (accepted->count >= max_rules) {
            continue;
        }
        rule->accepted = 1;
        accepted->items[accepted->count++] = *rule;
    }

    metrics->accepted_rules = accepted->count;
    metrics->accepted_conditional_rules = 0;
    metrics->accepted_gap_rules = 0;
    metrics->accepted_candidate_rules = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].accepted = 0;
    }
    for (size_t i = 0; i < accepted->count; ++i) {
        if (expanded_rule_is_conditional(&accepted->items[i])) {
            ++metrics->accepted_conditional_rules;
        }
        if (accepted->items[i].kind == RULE_FILTER_GAP_MOD) {
            ++metrics->accepted_gap_rules;
        } else {
            ++metrics->accepted_candidate_rules;
        }
        for (size_t candidate = 0; candidate < rules->count; ++candidate) {
            if (expanded_rule_same_key(&rules->items[candidate], &accepted->items[i])) {
                rules->items[candidate].accepted = 1;
                break;
            }
        }
    }
    return 1;
}

int expanded_rule_rejects(const ExpandedRule *rule,
                          uint32_t p,
                          uint32_t gap,
                          VerifierWork *work)
{
    uint32_t candidate = p + gap;
    if (expanded_rule_is_conditional(rule)) {
        ++work->mod_tests;
        if (p % rule->state_modulus != rule->state_residue) {
            return 0;
        }
    }
    ++work->mod_tests;
    if (candidate <= rule->filter_modulus) {
        return 0;
    }
    if (rule->kind == RULE_FILTER_GAP_MOD) {
        return gap % rule->filter_modulus == rule->filter_residue;
    }
    return candidate % rule->filter_modulus == rule->filter_residue;
}

VerifierWork expanded_rule_scan_work(const Sieve *sieve,
                                      const ExpandedRuleSet *rules,
                                      uint32_t p,
                                      uint32_t true_gap,
                                      int *false_rejection)
{
    VerifierWork work = {0, 0, 0};
    *false_rejection = 0;

    for (uint32_t gap = 2; gap <= true_gap; gap += 2) {
        int rejected = 0;
        ++work.candidates;
        for (size_t i = 0; i < rules->count; ++i) {
            if (expanded_rule_rejects(&rules->items[i], p, gap, &work)) {
                rejected = 1;
                break;
            }
        }
        if (rejected) {
            if (gap == true_gap) {
                *false_rejection = 1;
                return work;
            }
            continue;
        }
        ++work.exact_tests;
        if (is_prime_sieve(sieve, p + gap)) {
            return work;
        }
    }

    return work;
}

RuleSearchMetrics evaluate_expanded_rules(const Sieve *sieve,
                                          const ExpandedRuleSet *rules,
                                          const GapDataset *dataset,
                                          size_t max_states,
                                          RuleSearchMetrics metrics)
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
        VerifierWork expanded = expanded_rule_scan_work(sieve,
                                                        rules,
                                                        p,
                                                        gap,
                                                        &false_rejection);
        metrics.direct_exact_tests += (double)direct.exact_tests;
        metrics.hand_wheel_exact_tests += (double)hand.exact_tests;
        metrics.expanded_exact_tests += (double)expanded.exact_tests;
        metrics.expanded_mod_tests += (double)expanded.mod_tests;
        metrics.false_rejections += false_rejection ? 1u : 0u;
    }

    metrics.samples = samples;
    if (samples != 0) {
        double denominator = (double)samples;
        metrics.direct_exact_tests /= denominator;
        metrics.hand_wheel_exact_tests /= denominator;
        metrics.expanded_exact_tests /= denominator;
        metrics.expanded_mod_tests /= denominator;
    }
    return metrics;
}

PRIME_GAP_TEST_API int save_expanded_rule_checkpoint(const ExpandedRuleSet *rules,
                                                     const RuleSearchConfig *config,
                                                     const RuleSearchMetrics *metrics,
                                                     const char *path,
                                                     char out_hash[65])
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    int ok = fprintf(fp, "crlp.expanded_rules.v1\n") >= 0 &&
             fprintf(fp, "grammar_version %u\n", (uint32_t)RULE_SEARCH_GRAMMAR_VERSION) >= 0 &&
             fprintf(fp,
                     "include_state_conditions %u\n"
                     "include_candidate_filters %u\n"
                     "include_gap_filters %u\n",
                     config->include_state_conditions ? 1u : 0u,
                     config->include_candidate_filters ? 1u : 0u,
                     config->include_gap_filters ? 1u : 0u) >= 0 &&
             fprintf(fp, "moduli") >= 0;
    for (size_t i = 0; ok && i < config->mod_count; ++i) {
        ok = fprintf(fp, "%s%u", i == 0 ? " " : ",", config->moduli[i]) >= 0;
    }
    ok = ok && fprintf(fp,
                       "\nrule_count %zu\n"
                       "proposed_rules %zu\n"
                       "verified_rules %zu\n"
                       "accepted_rules %zu\n"
                       "false_rejections %zu\n"
                       "direct_exact %.17g\n"
                       "hand_wheel_exact %.17g\n"
                       "expanded_exact %.17g\n",
                       rules->count,
                       metrics->proposed_rules,
                       metrics->verified_rules,
                       metrics->accepted_rules,
                       metrics->false_rejections,
                       metrics->direct_exact_tests,
                       metrics->hand_wheel_exact_tests,
                       metrics->expanded_exact_tests) >= 0;

    for (size_t i = 0; ok && i < rules->count; ++i) {
        const ExpandedRule *rule = &rules->items[i];
        ok = fprintf(fp,
                     "rule %u %u %u %u %u %zu %zu %.9g ",
                     (uint32_t)rule->kind,
                     rule->state_modulus,
                     rule->state_residue,
                     rule->filter_modulus,
                     rule->filter_residue,
                     rule->train_rejects,
                     rule->validation_rejects,
                     (double)rule->model_score) >= 0;
        if (ok) {
            print_expanded_rule_text(fp, rule);
            ok = fputc('\n', fp) != EOF;
        }
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok) {
        return 0;
    }
    if (out_hash != NULL && !nn_hash_checkpoint_hex(path, out_hash)) {
        strcpy(out_hash, "unknown");
    }
    return 1;
}

static void hash_expanded_rule_set_hex(const ExpandedRuleSet *rules,
                                       const RuleSearchConfig *config,
                                       const RuleSearchMetrics *metrics,
                                       char out_hash[65])
{
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = mix_rule_hash_u64(hash, (uint64_t)RULE_SEARCH_GRAMMAR_VERSION);
    hash = mix_rule_hash_u64(hash, (uint64_t)config->include_state_conditions);
    hash = mix_rule_hash_u64(hash, (uint64_t)config->include_candidate_filters);
    hash = mix_rule_hash_u64(hash, (uint64_t)config->include_gap_filters);
    hash = mix_rule_hash_u64(hash, (uint64_t)config->max_rules);
    hash = mix_rule_hash_u64(hash, (uint64_t)config->proposer_mode);
    hash = mix_rule_hash_u64(hash, config->proposer_seed);
    if (config->proposer_mode == RULE_PROPOSER_CLOSED_LOOP) {
        hash = mix_rule_hash_u64(hash, (uint64_t)rule_search_effective_hidden1(config));
        hash = mix_rule_hash_u64(hash, (uint64_t)rule_search_effective_hidden2(config));
    } else if (config->proposer_mode == RULE_PROPOSER_UNIFIED) {
        hash = mix_rule_hash_u64(hash, (uint64_t)rule_search_effective_unified_hidden1(config));
        hash = mix_rule_hash_u64(hash, (uint64_t)rule_search_effective_unified_hidden2(config));
    }
    hash = mix_rule_hash_u64(hash, (uint64_t)metrics->accepted_rules);
    for (size_t i = 0; i < config->mod_count; ++i) {
        hash = mix_rule_hash_u64(hash, (uint64_t)config->moduli[i]);
    }
    for (size_t i = 0; i < rules->count; ++i) {
        hash = mix_rule_hash_u64(hash, expanded_rule_key_hash(&rules->items[i], 0u));
    }
    snprintf(out_hash, 65, "ruleset-%016llx", (unsigned long long)hash);
}

static void strip_line_ending(char *line)
{
    size_t length = strlen(line);
    while (length != 0 &&
           (line[length - 1u] == '\n' || line[length - 1u] == '\r')) {
        line[--length] = '\0';
    }
}

int load_expanded_rules_with_hash(const char *path,
                                  ExpandedRuleSet *rules,
                                  RuleSearchConfig *config,
                                  RuleSearchMetrics *metrics,
                                  char out_hash[65])
{
    memset(rules, 0, sizeof(*rules));
    if (config != NULL) {
        *config = default_rule_search_config();
        config->mod_count = 0;
    }
    if (metrics != NULL) {
        memset(metrics, 0, sizeof(*metrics));
    }
    if (path == NULL) {
        return 0;
    }

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    RuleSearchConfig parsed_config = default_rule_search_config();
    parsed_config.mod_count = 0;
    RuleSearchMetrics parsed_metrics;
    memset(&parsed_metrics, 0, sizeof(parsed_metrics));

    char line[4096];
    unsigned int grammar_version = 0;
    unsigned int include_state = 0;
    unsigned int include_candidate = 0;
    unsigned int include_gap = 0;
    size_t rule_count = 0;
    int ok = fgets(line, sizeof(line), fp) != NULL;
    if (ok) {
        strip_line_ending(line);
        ok = strcmp(line, "crlp.expanded_rules.v1") == 0;
    }
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "grammar_version %u", &grammar_version) == 1 &&
         grammar_version == (uint32_t)RULE_SEARCH_GRAMMAR_VERSION;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "include_state_conditions %u", &include_state) == 1 &&
         include_state <= 1u;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "include_candidate_filters %u", &include_candidate) == 1 &&
         include_candidate <= 1u;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "include_gap_filters %u", &include_gap) == 1 &&
         include_gap <= 1u && (include_candidate != 0 || include_gap != 0);
    if (ok) {
        parsed_config.include_state_conditions = include_state != 0;
        parsed_config.include_candidate_filters = include_candidate != 0;
        parsed_config.include_gap_filters = include_gap != 0;
    }
    ok = ok && fgets(line, sizeof(line), fp) != NULL;
    if (ok) {
        strip_line_ending(line);
        ok = strncmp(line, "moduli ", 7) == 0 &&
             parse_rule_search_moduli(line + 7,
                                      parsed_config.moduli,
                                      &parsed_config.mod_count);
    }
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "rule_count %zu", &rule_count) == 1;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "proposed_rules %zu", &parsed_metrics.proposed_rules) == 1;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "verified_rules %zu", &parsed_metrics.verified_rules) == 1;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "accepted_rules %zu", &parsed_metrics.accepted_rules) == 1;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "false_rejections %zu", &parsed_metrics.false_rejections) == 1;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "direct_exact %lf", &parsed_metrics.direct_exact_tests) == 1;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "hand_wheel_exact %lf", &parsed_metrics.hand_wheel_exact_tests) == 1;
    ok = ok && fgets(line, sizeof(line), fp) != NULL &&
         sscanf(line, "expanded_exact %lf", &parsed_metrics.expanded_exact_tests) == 1;

    if (ok) {
        rules->items = calloc(rule_count == 0 ? 1u : rule_count, sizeof(*rules->items));
        ok = rules->items != NULL;
    }

    for (size_t i = 0; ok && i < rule_count; ++i) {
        unsigned int kind = 0;
        unsigned int state_modulus = 0;
        unsigned int state_residue = 0;
        unsigned int filter_modulus = 0;
        unsigned int filter_residue = 0;
        size_t train_rejects = 0;
        size_t validation_rejects = 0;
        double model_score = 0.0;
        ok = fgets(line, sizeof(line), fp) != NULL &&
             sscanf(line,
                    "rule %u %u %u %u %u %zu %zu %lf",
                    &kind,
                    &state_modulus,
                    &state_residue,
                    &filter_modulus,
                    &filter_residue,
                    &train_rejects,
                    &validation_rejects,
                    &model_score) == 8 &&
             kind <= (uint32_t)RULE_FILTER_GAP_MOD &&
             state_modulus != 0 && state_residue < state_modulus &&
             filter_modulus >= 3 && filter_residue < filter_modulus;
        if (!ok) {
            break;
        }

        ExpandedRule *rule = &rules->items[i];
        rule->kind = (RuleFilterKind)kind;
        rule->state_modulus = (uint32_t)state_modulus;
        rule->state_residue = (uint32_t)state_residue;
        rule->filter_modulus = (uint32_t)filter_modulus;
        rule->filter_residue = (uint32_t)filter_residue;
        rule->train_rejects = train_rejects;
        rule->validation_rejects = validation_rejects;
        rule->model_score = (nn_real)model_score;
        rule->accepted = 1;
        ok = expanded_rule_has_deterministic_certificate(rule);
        if (ok) {
            ++rules->count;
            if (expanded_rule_is_conditional(rule)) {
                ++parsed_metrics.accepted_conditional_rules;
            }
            if (rule->kind == RULE_FILTER_GAP_MOD) {
                ++parsed_metrics.accepted_gap_rules;
            } else {
                ++parsed_metrics.accepted_candidate_rules;
            }
        }
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    ok = ok && rules->count == rule_count &&
         parsed_metrics.accepted_rules == rule_count;
    if (!ok) {
        free_expanded_rule_set(rules);
        return 0;
    }
    if (config != NULL) {
        *config = parsed_config;
    }
    if (metrics != NULL) {
        *metrics = parsed_metrics;
    }
    if (out_hash != NULL && !nn_hash_checkpoint_hex(path, out_hash)) {
        strcpy(out_hash, "unknown");
    }
    return 1;
}

static int write_expanded_rule_trace(const char *path,
                                     const ExpandedRuleSet *rules,
                                     const RuleSearchConfig *config)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    int ok = 1;
    for (size_t i = 0; ok && i < rules->count; ++i) {
        const ExpandedRule *rule = &rules->items[i];
        const char *result = "verified_not_selected";
        if (rule->train_false_rejections != 0 || rule->validation_false_rejections != 0) {
            result = "unsafe";
        } else if (rule->train_rejects == 0 && rule->validation_rejects == 0) {
            result = "no_support";
        } else if (!expanded_rule_has_deterministic_certificate(rule)) {
            result = "uncertified";
        } else if (rule->accepted) {
            result = "accepted";
        }

        char grammar_json[32];
        snprintf(grammar_json, sizeof(grammar_json), "%u", (uint32_t)RULE_SEARCH_GRAMMAR_VERSION);
        char rule_text[160];
        ok = format_expanded_rule_text(rule_text, sizeof(rule_text), rule);
        char extra_json[512];
        if (ok) {
            snprintf(extra_json,
                     sizeof(extra_json),
                     "\"state_modulus\":%u,"
                     "\"state_residue\":%u,"
                     "\"filter_modulus\":%u,"
                     "\"filter_residue\":%u,"
                     "\"conditional\":%s,"
                     "\"derived_divisibility\":%s,"
                     "\"deterministic_certificate\":%s,"
                     "\"state_conditions_enabled\":%s,"
                     "\"train_limit\":%zu,"
                     "\"test_limit\":%zu",
                     rule->state_modulus,
                     rule->state_residue,
                     rule->filter_modulus,
                     rule->filter_residue,
                     expanded_rule_is_conditional(rule) ? "true" : "false",
                     expanded_rule_is_derived_divisibility(rule) ? "true" : "false",
                     expanded_rule_has_deterministic_certificate(rule) ? "true" : "false",
                     config->include_state_conditions ? "true" : "false",
                     config->train_limit,
                     config->test_limit);
            NNRuleTraceRecord record = {
                .trace_version = 1,
                .task_name = "prime_gap",
                .grammar_version_json = grammar_json,
                .selection_strategy = rule_proposer_mode_name(config->proposer_mode),
                .proposer_feature_set = rule_proposer_feature_set_name(config->proposer_mode),
                .proposer_target = rule_proposer_target_name(config->proposer_mode),
                .rule_id = i,
                .rule_rank = i,
                .rule_kind = rule_filter_kind_name(rule->kind),
                .accepted = rule->accepted,
                .verifier_result = result,
                .certificate = expanded_rule_has_deterministic_certificate(rule)
                                   ? "deterministic_modular_divisibility"
                                   : "none",
                .model_score = (double)rule->model_score,
                .utility = rule->utility,
                .train_rejects = rule->train_rejects,
                .train_false_rejections = rule->train_false_rejections,
                .validation_rejects = rule->validation_rejects,
                .validation_false_rejections = rule->validation_false_rejections,
                .train_marginal_utility = rule->train_marginal_utility,
                .validation_marginal_utility = rule->validation_marginal_utility,
                .rule_text = rule_text,
                .extra_json = extra_json,
            };
            ok = nn_rule_trace_write_json(fp, &record);
        }
    }
    if (fclose(fp) != 0) {
        ok = 0;
    }
    return ok;
}

static void print_rule_search_report(const ExpandedRuleSet *accepted,
                                     const RuleSearchConfig *config,
                                     const RuleSearchMetrics *metrics,
                                     const char *rule_hash)
{
    double hand_saved = metrics->direct_exact_tests - metrics->hand_wheel_exact_tests;
    double expanded_saved = metrics->direct_exact_tests - metrics->expanded_exact_tests;
    double hand_saved_pct = metrics->direct_exact_tests == 0.0
                                ? 0.0
                                : 100.0 * hand_saved / metrics->direct_exact_tests;
    double expanded_saved_pct = metrics->direct_exact_tests == 0.0
                                    ? 0.0
                                    : 100.0 * expanded_saved / metrics->direct_exact_tests;
    char closed_loop_hidden_json[64];
    rule_search_closed_loop_hidden_json(config,
                                        closed_loop_hidden_json,
                                        sizeof(closed_loop_hidden_json));
    char unified_hidden_json[64];
    rule_search_unified_hidden_json(config,
                                    unified_hidden_json,
                                    sizeof(unified_hidden_json));

    char grammar_json[32];
    snprintf(grammar_json, sizeof(grammar_json), "%u", (uint32_t)RULE_SEARCH_GRAMMAR_VERSION);
    char extra_json[512];
    snprintf(extra_json,
             sizeof(extra_json),
             "\"include_state_conditions\":%s,"
             "\"include_candidate_filters\":%s,"
             "\"include_gap_filters\":%s,"
             "\"accepted_conditional_rules\":%zu,"
             "\"accepted_candidate_rules\":%zu,"
             "\"accepted_gap_rules\":%zu",
             config->include_state_conditions ? "true" : "false",
             config->include_candidate_filters ? "true" : "false",
             config->include_gap_filters ? "true" : "false",
             metrics->accepted_conditional_rules,
             metrics->accepted_candidate_rules,
             metrics->accepted_gap_rules);
    NNRuleSearchSummary summary = {
        .task_name = "prime_gap",
        .grammar_version_json = grammar_json,
        .rule_set_hash = rule_hash == NULL ? "unknown" : rule_hash,
        .train_limit = config->train_limit,
        .test_limit = config->test_limit,
        .sample_count = metrics->samples,
        .max_rules = config->max_rules,
        .max_marginal_states = config->max_marginal_states,
        .softmax_group_size = metrics->softmax_group_size,
        .closed_loop_hidden_json = closed_loop_hidden_json,
        .unified_scorer_hidden_json = unified_hidden_json,
        .selection_strategy = rule_proposer_mode_name(config->proposer_mode),
        .selection_seed = config->proposer_seed,
        .proposed_rules = metrics->proposed_rules,
        .proposer_calls = metrics->proposer_calls,
        .verified_rules = metrics->verified_rules,
        .unsafe_rules = metrics->unsafe_rules,
        .no_support_rules = metrics->no_support_rules,
        .uncertified_rules = metrics->uncertified_rules,
        .accepted_rules = metrics->accepted_rules,
        .false_rejections = metrics->false_rejections,
        .proposer_feature_set = rule_proposer_feature_set_name(config->proposer_mode),
        .proposer_target = rule_proposer_target_name(config->proposer_mode),
        .residual_train_samples = metrics->residual_train_samples,
        .train_marginal_samples = metrics->train_marginal_samples,
        .validation_marginal_samples = metrics->validation_marginal_samples,
        .train_marginal_events = metrics->train_marginal_events,
        .validation_marginal_events = metrics->validation_marginal_events,
        .coverage_cache_hits = metrics->coverage_cache_hits,
        .coverage_cache_misses = metrics->coverage_cache_misses,
        .coverage_cache_writes = metrics->coverage_cache_writes,
        .parameters = metrics->parameters,
        .forward_flops = metrics->forward_flops,
        .unified_action_updates = metrics->unified_action_updates,
        .unified_rule_updates = metrics->unified_rule_updates,
        .proposer_loss = metrics->proposer_loss,
        .direct_exact = metrics->direct_exact_tests,
        .hand_wheel_exact = metrics->hand_wheel_exact_tests,
        .expanded_exact = metrics->expanded_exact_tests,
        .expanded_mod = metrics->expanded_mod_tests,
        .expanded_saved_pct = expanded_saved_pct,
        .extra_json = extra_json,
    };
    nn_rule_search_summary_write_json(stdout, &summary);

    printf("\n[expanded prime rule search]\n");
    printf("  grammar_version=%u moduli=", (uint32_t)RULE_SEARCH_GRAMMAR_VERSION);
    for (size_t i = 0; i < config->mod_count; ++i) {
        printf("%s%u", i == 0 ? "" : ",", config->moduli[i]);
    }
    printf("\n");
    printf("  grammar: %s%s%s filters %s p %% m conditions\n",
           config->include_candidate_filters ? "candidate" : "",
           config->include_candidate_filters && config->include_gap_filters ? " + " : "",
           config->include_gap_filters ? "gap" : "",
           config->include_state_conditions ? "with optional" : "without");
    printf("  selection_strategy=%s max_rules=%zu max_marginal_states=%zu seed=%llu\n",
           rule_proposer_mode_name(config->proposer_mode),
           config->max_rules,
           config->max_marginal_states,
           (unsigned long long)config->proposer_seed);
    if (metrics->softmax_group_size != 0) {
        printf("  softmax_group_size=%zu\n", metrics->softmax_group_size);
    }
    printf("  proposer_target=%s marginal_samples train=%zu validation=%zu events train=%zu validation=%zu\n",
           rule_proposer_target_name(config->proposer_mode),
           metrics->train_marginal_samples,
           metrics->validation_marginal_samples,
           metrics->train_marginal_events,
           metrics->validation_marginal_events);
    printf("  proposed=%zu calls=%zu verified=%zu unsafe=%zu no_support=%zu uncertified=%zu accepted=%zu\n",
           metrics->proposed_rules,
           metrics->proposer_calls,
           metrics->verified_rules,
           metrics->unsafe_rules,
           metrics->no_support_rules,
           metrics->uncertified_rules,
           metrics->accepted_rules);
    printf("  accepted conditional=%zu gap_rules=%zu candidate_rules=%zu\n",
           metrics->accepted_conditional_rules,
           metrics->accepted_gap_rules,
           metrics->accepted_candidate_rules);
    if (config->proposer_mode == RULE_PROPOSER_UNIFIED) {
        printf("  unified_transfer feature_set=%s hidden=%zux%zu parameters=%zu "
               "forward_flops=%zu action_updates=%zu rule_updates=%zu "
               "rule_ce=%.6f residual_rows=%zu\n",
               rule_proposer_feature_set_name(config->proposer_mode),
               rule_search_effective_unified_hidden1(config),
               rule_search_effective_unified_hidden2(config),
               metrics->parameters,
               metrics->forward_flops,
               metrics->unified_action_updates,
               metrics->unified_rule_updates,
               metrics->proposer_loss,
               metrics->residual_train_samples);
    } else if (config->proposer_mode == RULE_PROPOSER_NEURAL ||
               config->proposer_mode == RULE_PROPOSER_CLOSED_LOOP) {
        printf("  neural_rule_scorer_epochs=%u feature_set=%s hidden=%zux%zu "
               "parameters=%zu forward_flops=%zu final_mse=%.6f residual_samples=%zu\n",
               (uint32_t)RULE_SEARCH_SCORER_EPOCHS,
               rule_proposer_feature_set_name(config->proposer_mode),
               config->proposer_mode == RULE_PROPOSER_CLOSED_LOOP
                   ? rule_search_effective_hidden1(config)
                   : (config->proposer_mode == RULE_PROPOSER_UNIFIED
                          ? rule_search_effective_unified_hidden1(config)
                          : 20u),
               config->proposer_mode == RULE_PROPOSER_CLOSED_LOOP
                   ? rule_search_effective_hidden2(config)
                   : (config->proposer_mode == RULE_PROPOSER_UNIFIED
                          ? rule_search_effective_unified_hidden2(config)
                          : 10u),
               metrics->parameters,
               metrics->forward_flops,
               metrics->proposer_loss,
               metrics->residual_train_samples);
    } else {
        printf("  trained_rule_scorer=none feature_set=%s\n",
               rule_proposer_feature_set_name(config->proposer_mode));
    }
    printf("  verifier diagnostics samples=%zu false_rejections=%zu\n",
           metrics->samples,
           metrics->false_rejections);
    printf("    direct exact=%.2f\n", metrics->direct_exact_tests);
    printf("    hand wheel exact=%.2f saved=%.2f (%.1f%%)\n",
           metrics->hand_wheel_exact_tests,
           hand_saved,
           hand_saved_pct);
    printf("    expanded exact=%.2f saved=%.2f (%.1f%%) mod_tests=%.2f\n",
           metrics->expanded_exact_tests,
           expanded_saved,
           expanded_saved_pct,
           metrics->expanded_mod_tests);
    printf("  top accepted rules");
    for (size_t i = 0; i < accepted->count && i < 12; ++i) {
        printf("%s", i == 0 ? ": " : ", ");
        print_expanded_rule_text(stdout, &accepted->items[i]);
    }
    if (accepted->count == 0) {
        printf(": none");
    } else if (accepted->count > 12) {
        printf(", ...");
    }
    printf("\n");
}

int nn_prime_rule_search_main(int argc, char **argv)
{
    memset(&RULE_COVERAGE_CACHE_STATS, 0, sizeof(RULE_COVERAGE_CACHE_STATS));

    RuleSearchConfig config;
    if (!parse_rule_search_config(argc, argv, &config)) {
        print_rule_search_usage(stderr);
        return EXIT_FAILURE;
    }

    uint32_t train_limit = (uint32_t)config.train_limit;
    uint32_t test_limit = (uint32_t)config.test_limit;
    uint32_t sieve_limit = test_limit + (uint32_t)G_MAX;
    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    if (!build_sieve(sieve_limit, &sieve) ||
        !build_gap_datasets(&sieve, train_limit, test_limit, &train, &validation, &test)) {
        fputs("failed to build prime-gap rule-search data\n", stderr);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    RuleSearchGrammar grammar;
    ExpandedRuleSet candidates;
    ExpandedRuleSet accepted;
    RuleSearchMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    memset(&grammar, 0, sizeof(grammar));
    memset(&candidates, 0, sizeof(candidates));
    memset(&accepted, 0, sizeof(accepted));

    int ok = build_rule_search_grammar(&config, &grammar) &&
             build_expanded_rules(&grammar, &candidates);
    if (ok) {
        accumulate_expanded_rule_stats(&candidates,
                                       &config,
                                       &grammar,
                                       &train,
                                       config.max_train_states,
                                       0);
        accumulate_expanded_rule_stats(&candidates,
                                       &config,
                                       &grammar,
                                       &validation,
                                       config.max_train_states,
                                       1);
        size_t marginal_state_limit = rule_search_marginal_state_limit(&config);
        ok = compute_marginal_rule_utilities(&candidates,
                                             &config,
                                             &grammar,
                                             &train,
                                             marginal_state_limit,
                                             0,
                                             &metrics.train_marginal_samples,
                                             &metrics.train_marginal_events) &&
             compute_marginal_rule_utilities(&candidates,
                                             &config,
                                             &grammar,
                                             &validation,
                                             marginal_state_limit,
                                             1,
                                             &metrics.validation_marginal_samples,
                                             &metrics.validation_marginal_events);
        if (ok && config.proposer_mode == RULE_PROPOSER_CLOSED_LOOP) {
            set_expanded_rule_utilities(&candidates);
            ok = select_expanded_rules_closed_loop(&candidates,
                                                   &config,
                                                   &grammar,
                                                   &train,
                                                   marginal_state_limit,
                                                   &accepted,
                                                   &metrics);
        } else if (ok && config.proposer_mode == RULE_PROPOSER_UNIFIED) {
            set_expanded_rule_utilities(&candidates);
            ok = select_expanded_rules_unified(&candidates,
                                               &config,
                                               &grammar,
                                               &train,
                                               marginal_state_limit,
                                               &accepted,
                                               &metrics);
        } else if (ok) {
            ok = score_expanded_rules(&candidates, &config, &metrics) &&
                 select_expanded_rules(&candidates, config.max_rules, &accepted, &metrics);
        }
    }

    char rule_hash[65] = "unknown";
    if (ok) {
        metrics = evaluate_expanded_rules(&sieve,
                                          &accepted,
                                          &test,
                                          config.max_test_states,
                                          metrics);
        metrics.coverage_cache_hits = RULE_COVERAGE_CACHE_STATS.hits;
        metrics.coverage_cache_misses = RULE_COVERAGE_CACHE_STATS.misses;
        metrics.coverage_cache_writes = RULE_COVERAGE_CACHE_STATS.writes;
        if (config.save_path != NULL) {
            ok = save_expanded_rule_checkpoint(&accepted,
                                               &config,
                                               &metrics,
                                               config.save_path,
                                               rule_hash);
        } else {
            hash_expanded_rule_set_hex(&accepted, &config, &metrics, rule_hash);
        }
    }
    if (ok && config.trace_path != NULL) {
        ok = write_expanded_rule_trace(config.trace_path, &candidates, &config);
    }

    if (ok) {
        print_rule_search_report(&accepted, &config, &metrics, rule_hash);
        if (config.save_path != NULL) {
            printf("saved expanded rules: %s hash=%s\n", config.save_path, rule_hash);
        }
        if (config.trace_path != NULL) {
            printf("wrote rule trace: %s rows=%zu\n", config.trace_path, candidates.count);
        }
    }

    free_expanded_rule_set(&accepted);
    free_expanded_rule_set(&candidates);
    free_rule_search_grammar(&grammar);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
