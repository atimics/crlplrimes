/* Expanded rule grammar, verifier-gated rule selection, trace writing, and CLI entry point. */

#include "internal.h"

static RuleSearchConfig default_rule_search_config(void)
{
    RuleSearchConfig config;
    memset(&config, 0, sizeof(config));
    config.train_limit = DEFAULT_TRAIN_LIMIT;
    config.test_limit = DEFAULT_TEST_LIMIT;
    config.max_train_states = DEFAULT_MAX_TRAIN_STATES;
    config.max_test_states = DEFAULT_MAX_TEST_STATES;
    config.max_marginal_states = (size_t)RULE_SEARCH_MARGINAL_MAX_STATES;
    config.max_rules = 256;
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
    }
    return "unknown";
}

static const char *rule_proposer_feature_set_name(RuleProposerMode mode)
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
    }
    return "unknown";
}

static const char *rule_proposer_target_name(RuleProposerMode mode)
{
    switch (mode) {
    case RULE_PROPOSER_NEURAL:
        return "train_marginal_utility";
    case RULE_PROPOSER_CLOSED_LOOP:
        return "residual_marginal_utility";
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

static int parse_rule_search_config(int argc, char **argv, RuleSearchConfig *config)
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
          "[--proposer neural|closed-loop|random|grammar|structural|validation-oracle] "
          "[--proposer-seed n] "
          "[--gap-only|--candidate-only] [--unconditional-only] "
          "[--moduli 3,5,7,...] [--moduli-range 3:255] "
          "[--trace path] [--save path]\n",
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

static void print_expanded_rule_text(FILE *fp, const ExpandedRule *rule)
{
    if (expanded_rule_is_conditional(rule)) {
        fprintf(fp,
                "if p %% %u == %u then reject %s %% %u == %u",
                rule->state_modulus,
                rule->state_residue,
                rule->kind == RULE_FILTER_GAP_MOD ? "gap" : "candidate",
                rule->filter_modulus,
                rule->filter_residue);
    } else {
        fprintf(fp,
                "reject %s %% %u == %u",
                rule->kind == RULE_FILTER_GAP_MOD ? "gap" : "candidate",
                rule->filter_modulus,
                rule->filter_residue);
    }
}

static void free_rule_search_grammar(RuleSearchGrammar *grammar)
{
    free(grammar->state_atoms);
    free(grammar->filter_atoms);
    free(grammar->state_offsets);
    free(grammar->candidate_offsets);
    free(grammar->gap_offsets);
    memset(grammar, 0, sizeof(*grammar));
}

static int build_rule_search_grammar(const RuleSearchConfig *config,
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

static int build_expanded_rules(const RuleSearchGrammar *grammar, ExpandedRuleSet *rules)
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

static void accumulate_expanded_rule_stats(ExpandedRuleSet *rules,
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

static size_t rule_search_marginal_state_limit(const RuleSearchConfig *config)
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
    if (!nn_rule_coverage_init(coverage, rules->count, events)) {
        return 0;
    }
    if (events == 0 || rules->count == 0) {
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

static int compute_marginal_rule_utilities(ExpandedRuleSet *rules,
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

static void fill_expanded_rule_features(const ExpandedRule *rule,
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

static void set_expanded_rule_utilities(ExpandedRuleSet *rules)
{
    for (size_t i = 0; i < rules->count; ++i) {
        ExpandedRule *rule = &rules->items[i];
        rule->utility = rule->validation_false_rejections == 0
                            ? rule->validation_marginal_utility
                            : -(double)rule->validation_false_rejections;
    }
}

typedef struct {
    nn_real values[RULE_SEARCH_FEATURE_COUNT];
} RuleProposerInputs;

typedef struct {
    size_t train_rejects;
    size_t train_false_rejections;
    double train_marginal_utility;
    nn_real marginal_target;
} RuleProposerTargets;

typedef struct {
    RuleProposerInputs inputs;
    RuleProposerTargets targets;
} RuleProposerSample;

static void fill_rule_proposer_sample(const ExpandedRule *rule,
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
                                           double *loss_out)
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
                                double *loss_out)
{
    if (config->proposer_mode == RULE_PROPOSER_NEURAL) {
        return score_expanded_rules_with_model(rules, config, loss_out);
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

static int select_expanded_rules_closed_loop(ExpandedRuleSet *rules,
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

    NNClosedLoopRulePolicyConfig policy_config;
    nn_closed_loop_rule_policy_config_defaults(&policy_config,
                                               (size_t)RULE_DISCOVERY_FEATURE_COUNT);
    policy_config.max_rules = config->max_rules;
    policy_config.max_train_rules_per_step = (size_t)RULE_DISCOVERY_MAX_TRAIN_RULES_PER_STEP;
    policy_config.epochs = (size_t)RULE_SEARCH_SCORER_EPOCHS;
    policy_config.seed = config->proposer_seed ^ 119417u;
    policy_config.verbose = 1;

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

static int save_expanded_rule_checkpoint(const ExpandedRuleSet *rules,
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

        ok = fprintf(fp,
                     "{\"schema\":\"crlp.rule_trace.v1\",\"trace_version\":1,"
                     "\"task\":\"prime_gap\",\"grammar_version\":%u,"
                     "\"rule_rank\":%zu,\"rule_kind\":\"%s\","
                     "\"state_modulus\":%u,\"state_residue\":%u,"
                     "\"filter_modulus\":%u,\"filter_residue\":%u,"
                     "\"conditional\":%s,\"derived_divisibility\":%s,"
                     "\"deterministic_certificate\":%s,"
                     "\"selection_strategy\":\"%s\","
                     "\"proposer_feature_set\":\"%s\","
                     "\"proposer_target\":\"%s\","
                     "\"state_conditions_enabled\":%s,"
                     "\"train_limit\":%zu,\"test_limit\":%zu,"
                     "\"train_rejects\":%zu,\"train_false_rejections\":%zu,"
                     "\"validation_rejects\":%zu,"
                     "\"validation_false_rejections\":%zu,"
                     "\"train_marginal_utility\":%.9g,"
                     "\"validation_marginal_utility\":%.9g,"
                     "\"model_score\":%.9g,\"utility\":%.9g,"
                     "\"accepted\":%s,\"verifier_result\":\"%s\","
                     "\"rule_text\":\"",
                     (uint32_t)RULE_SEARCH_GRAMMAR_VERSION,
                     i,
                     rule_filter_kind_name(rule->kind),
                     rule->state_modulus,
                     rule->state_residue,
                     rule->filter_modulus,
                     rule->filter_residue,
                     expanded_rule_is_conditional(rule) ? "true" : "false",
                     expanded_rule_is_derived_divisibility(rule) ? "true" : "false",
                     expanded_rule_has_deterministic_certificate(rule) ? "true" : "false",
                     rule_proposer_mode_name(config->proposer_mode),
                     rule_proposer_feature_set_name(config->proposer_mode),
                     rule_proposer_target_name(config->proposer_mode),
                     config->include_state_conditions ? "true" : "false",
                     config->train_limit,
                     config->test_limit,
                     rule->train_rejects,
                     rule->train_false_rejections,
                     rule->validation_rejects,
                     rule->validation_false_rejections,
                     rule->train_marginal_utility,
                     rule->validation_marginal_utility,
                     (double)rule->model_score,
                     rule->utility,
                     rule->accepted ? "true" : "false",
                     result) >= 0;
        if (ok) {
            print_expanded_rule_text(fp, rule);
            ok = fprintf(fp, "\"}\n") >= 0;
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

    printf("{\"schema\":\"crlp.rule_search.summary.v1\",\"task\":\"prime_gap\","
           "\"grammar_version\":%u,\"rule_set_hash\":\"%s\","
           "\"train_limit\":%zu,\"test_limit\":%zu,\"sample_count\":%zu,"
           "\"include_state_conditions\":%s,"
           "\"include_candidate_filters\":%s,\"include_gap_filters\":%s,"
           "\"max_rules\":%zu,\"max_marginal_states\":%zu,"
           "\"selection_strategy\":\"%s\","
           "\"selection_seed\":%llu,"
           "\"proposed_rules\":%zu,\"proposer_calls\":%zu,"
           "\"verified_rules\":%zu,"
           "\"unsafe_rules\":%zu,\"no_support_rules\":%zu,"
           "\"uncertified_rules\":%zu,"
           "\"accepted_rules\":%zu,\"accepted_conditional_rules\":%zu,"
           "\"accepted_candidate_rules\":%zu,\"accepted_gap_rules\":%zu,"
           "\"false_rejections\":%zu,\"proposer_feature_set\":\"%s\","
           "\"proposer_target\":\"%s\","
           "\"residual_train_samples\":%zu,"
           "\"train_marginal_samples\":%zu,"
           "\"validation_marginal_samples\":%zu,"
           "\"train_marginal_events\":%zu,"
           "\"validation_marginal_events\":%zu,"
           "\"proposer_loss\":%.8f,"
           "\"direct_exact\":%.8f,\"hand_wheel_exact\":%.8f,"
           "\"expanded_exact\":%.8f,\"expanded_mod\":%.8f,"
           "\"expanded_saved_pct\":%.8f}\n",
           (uint32_t)RULE_SEARCH_GRAMMAR_VERSION,
           rule_hash == NULL ? "unknown" : rule_hash,
           config->train_limit,
           config->test_limit,
           metrics->samples,
           config->include_state_conditions ? "true" : "false",
           config->include_candidate_filters ? "true" : "false",
           config->include_gap_filters ? "true" : "false",
           config->max_rules,
           config->max_marginal_states,
           rule_proposer_mode_name(config->proposer_mode),
           (unsigned long long)config->proposer_seed,
           metrics->proposed_rules,
           metrics->proposer_calls,
           metrics->verified_rules,
           metrics->unsafe_rules,
           metrics->no_support_rules,
           metrics->uncertified_rules,
           metrics->accepted_rules,
           metrics->accepted_conditional_rules,
           metrics->accepted_candidate_rules,
           metrics->accepted_gap_rules,
           metrics->false_rejections,
           rule_proposer_feature_set_name(config->proposer_mode),
           rule_proposer_target_name(config->proposer_mode),
           metrics->residual_train_samples,
           metrics->train_marginal_samples,
           metrics->validation_marginal_samples,
           metrics->train_marginal_events,
           metrics->validation_marginal_events,
           metrics->proposer_loss,
           metrics->direct_exact_tests,
           metrics->hand_wheel_exact_tests,
           metrics->expanded_exact_tests,
           metrics->expanded_mod_tests,
           expanded_saved_pct);

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
    if (config->proposer_mode == RULE_PROPOSER_NEURAL ||
        config->proposer_mode == RULE_PROPOSER_CLOSED_LOOP) {
        printf("  neural_rule_scorer_epochs=%u feature_set=%s final_mse=%.6f residual_samples=%zu\n",
               (uint32_t)RULE_SEARCH_SCORER_EPOCHS,
               rule_proposer_feature_set_name(config->proposer_mode),
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
        } else if (ok) {
            ok = score_expanded_rules(&candidates, &config, &metrics.proposer_loss) &&
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
