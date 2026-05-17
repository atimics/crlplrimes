#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    TASK_PRIME_CONSTELLATION = 0,
    TASK_KFREE = 1,
    TASK_FIBONACCI_LUCAS = 2,
    TASK_COUNT = 3
} ModularTask;

typedef enum {
    RULE_TUPLE_DIVISIBLE = 0,
    RULE_KFREE_DIVISIBLE = 1,
    RULE_RECURRENCE_ZERO = 2
} ModularRuleKind;

typedef enum {
    SEQUENCE_NONE = 0,
    SEQUENCE_FIBONACCI = 1,
    SEQUENCE_LUCAS = 2
} RecurrenceSequence;

typedef enum {
    SPLIT_TRAIN = 0,
    SPLIT_VALIDATION = 1,
    SPLIT_TEST = 2
} ModularSplit;

typedef enum {
    STRATEGY_GRAMMAR_ORDER = 0,
    STRATEGY_RANDOM = 1,
    STRATEGY_STRUCTURAL = 2,
    STRATEGY_LEARNED_STATIC = 3,
    STRATEGY_TRAIN_MARGINAL_ORACLE = 4,
    STRATEGY_VALIDATION_MARGINAL_ORACLE = 5,
    STRATEGY_COUNT = 6
} ModularSelectionStrategy;

typedef struct {
    ModularTask task;
    ModularRuleKind kind;
    RecurrenceSequence sequence;
    uint32_t prime;
    uint32_t power;
    uint32_t modulus;
    uint32_t residue;
    uint32_t tuple_offset;
    int certified;
    size_t train_support;
    size_t train_false_rejections;
    size_t train_marginal;
    size_t validation_support;
    size_t validation_marginal;
    size_t validation_false_rejections;
    size_t rule_index;
    int accepted;
    size_t accepted_rank;
    char text[128];
} ModularRule;

typedef struct {
    size_t max_states;
    size_t max_actions;
    size_t max_rules;
    ModularTask task;
    int run_all;
    uint64_t selection_seed;
    const char *trace_path;
} ModularGateConfig;

typedef struct {
    size_t proposed_rules;
    size_t certified_rules;
    size_t accepted_rules;
    size_t states;
    size_t states_with_target;
    size_t baseline_candidates;
    size_t gated_candidates;
    size_t rejected_candidates;
    size_t false_rejections;
} ModularGateSummary;

enum { LEARNED_FEATURE_COUNT = 8 };

typedef struct {
    double weights[LEARNED_FEATURE_COUNT];
    double loss;
} LearnedProposer;

static const uint32_t PRIME_RULE_PRIMES[] = {
    3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u,
};

static const uint32_t KFREE_RULE_PRIMES[] = {
    2u, 3u, 5u, 7u, 11u, 13u, 17u, 19u, 23u, 29u, 31u,
};

static const uint32_t RECURRENCE_RULE_PRIMES[] = {
    2u, 3u, 5u, 7u, 11u, 13u,
};

static const char *task_name(ModularTask task)
{
    switch (task) {
    case TASK_PRIME_CONSTELLATION:
        return "prime_constellation";
    case TASK_KFREE:
        return "kfree_squarefree";
    case TASK_FIBONACCI_LUCAS:
        return "fibonacci_lucas";
    default:
        return "unknown";
    }
}

static const char *certificate_name(ModularTask task)
{
    switch (task) {
    case TASK_PRIME_CONSTELLATION:
        return "tuple_member_composite_by_divisibility";
    case TASK_KFREE:
        return "power_divisibility_implies_not_squarefree";
    case TASK_FIBONACCI_LUCAS:
        return "pisano_period_zero_residue";
    default:
        return "unknown";
    }
}

static const char *sequence_name(RecurrenceSequence sequence)
{
    switch (sequence) {
    case SEQUENCE_FIBONACCI:
        return "fibonacci";
    case SEQUENCE_LUCAS:
        return "lucas";
    default:
        return "none";
    }
}

static const char *rule_kind_name(ModularRuleKind kind)
{
    switch (kind) {
    case RULE_TUPLE_DIVISIBLE:
        return "tuple_divisible";
    case RULE_KFREE_DIVISIBLE:
        return "kfree_divisible";
    case RULE_RECURRENCE_ZERO:
        return "recurrence_zero";
    default:
        return "unknown";
    }
}

static const char *strategy_name(ModularSelectionStrategy strategy)
{
    switch (strategy) {
    case STRATEGY_GRAMMAR_ORDER:
        return "grammar_order";
    case STRATEGY_RANDOM:
        return "random";
    case STRATEGY_STRUCTURAL:
        return "structural_heuristic";
    case STRATEGY_LEARNED_STATIC:
        return "learned_structural";
    case STRATEGY_TRAIN_MARGINAL_ORACLE:
        return "train_marginal_oracle";
    case STRATEGY_VALIDATION_MARGINAL_ORACLE:
        return "validation_marginal_oracle";
    default:
        return "unknown";
    }
}

static const char *strategy_feature_set(ModularSelectionStrategy strategy)
{
    switch (strategy) {
    case STRATEGY_RANDOM:
    case STRATEGY_GRAMMAR_ORDER:
        return "none";
    case STRATEGY_VALIDATION_MARGINAL_ORACLE:
        return "validation_counts";
    case STRATEGY_TRAIN_MARGINAL_ORACLE:
        return "train_counts";
    case STRATEGY_STRUCTURAL:
        return "structural_heuristic";
    case STRATEGY_LEARNED_STATIC:
        return "structural";
    default:
        return "unknown";
    }
}

static const char *strategy_target(ModularSelectionStrategy strategy)
{
    switch (strategy) {
    case STRATEGY_LEARNED_STATIC:
        return "train_rule_coverage";
    case STRATEGY_TRAIN_MARGINAL_ORACLE:
        return "train_marginal_utility";
    case STRATEGY_VALIDATION_MARGINAL_ORACLE:
        return "validation_marginal_utility";
    default:
        return "none";
    }
}

static uint64_t state_base(ModularTask task, size_t index, ModularSplit split)
{
    uint64_t offset = (uint64_t)index;
    switch (task) {
    case TASK_PRIME_CONSTELLATION:
        return (split == SPLIT_TEST ? 1009u : (split == SPLIT_VALIDATION ? 509u : 101u)) + offset;
    case TASK_KFREE:
        return (split == SPLIT_TEST ? 5000u : (split == SPLIT_VALIDATION ? 3000u : 1000u)) + offset;
    case TASK_FIBONACCI_LUCAS:
        return (split == SPLIT_TEST ? 1000u : (split == SPLIT_VALIDATION ? 500u : 10u)) + offset;
    default:
        return offset;
    }
}

static int is_prime_u64(uint64_t value)
{
    if (value < 2u) {
        return 0;
    }
    if (value == 2u) {
        return 1;
    }
    if ((value % 2u) == 0u) {
        return 0;
    }
    for (uint64_t d = 3u; d * d <= value; d += 2u) {
        if ((value % d) == 0u) {
            return 0;
        }
    }
    return 1;
}

static uint32_t pow_u32(uint32_t base, uint32_t power)
{
    uint32_t result = 1u;
    for (uint32_t i = 0u; i < power; ++i) {
        result *= base;
    }
    return result;
}

static int is_squarefree_u64(uint64_t value)
{
    for (uint64_t p = 2u; p * p <= value; ++p) {
        uint64_t square = p * p;
        if ((value % square) == 0u) {
            return 0;
        }
    }
    return 1;
}

static uint32_t fib_mod_u32(uint64_t index, uint32_t modulus)
{
    if (modulus == 1u) {
        return 0u;
    }
    uint32_t a = 0u;
    uint32_t b = 1u % modulus;
    for (uint64_t i = 0u; i < index; ++i) {
        uint32_t next = (uint32_t)(((uint64_t)a + b) % modulus);
        a = b;
        b = next;
    }
    return a;
}

static uint32_t lucas_mod_u32(uint64_t index, uint32_t modulus)
{
    if (modulus == 1u) {
        return 0u;
    }
    uint32_t a = 2u % modulus;
    uint32_t b = 1u % modulus;
    if (index == 0u) {
        return a;
    }
    for (uint64_t i = 1u; i < index; ++i) {
        uint32_t next = (uint32_t)(((uint64_t)a + b) % modulus);
        a = b;
        b = next;
    }
    return b;
}

static uint32_t sequence_mod_u32(RecurrenceSequence sequence, uint64_t index, uint32_t modulus)
{
    return sequence == SEQUENCE_LUCAS ? lucas_mod_u32(index, modulus)
                                      : fib_mod_u32(index, modulus);
}

static uint32_t pisano_period_u32(uint32_t modulus)
{
    uint32_t previous = 0u;
    uint32_t current = 1u % modulus;
    uint32_t limit = 6u * modulus * modulus + 16u;
    for (uint32_t period = 1u; period <= limit; ++period) {
        uint32_t next = (uint32_t)(((uint64_t)previous + current) % modulus);
        previous = current;
        current = next;
        if (previous == 0u && current == (1u % modulus)) {
            return period;
        }
    }
    return 0u;
}

static int recurrence_candidate_is_clean(uint64_t index)
{
    for (size_t i = 0u; i < sizeof(RECURRENCE_RULE_PRIMES) / sizeof(RECURRENCE_RULE_PRIMES[0]); ++i) {
        uint32_t p = RECURRENCE_RULE_PRIMES[i];
        if (fib_mod_u32(index, p) == 0u || lucas_mod_u32(index, p) == 0u) {
            return 0;
        }
    }
    return 1;
}

static int candidate_is_valid(ModularTask task, uint64_t state, size_t action)
{
    uint64_t candidate = state + (uint64_t)action;
    switch (task) {
    case TASK_PRIME_CONSTELLATION:
        return is_prime_u64(candidate) && is_prime_u64(candidate + 2u);
    case TASK_KFREE:
        return is_squarefree_u64(candidate);
    case TASK_FIBONACCI_LUCAS:
        return recurrence_candidate_is_clean(candidate);
    default:
        return 0;
    }
}

static size_t true_action_for_state(ModularTask task, uint64_t state, size_t max_actions)
{
    for (size_t action = 0u; action < max_actions; ++action) {
        if (candidate_is_valid(task, state, action)) {
            return action;
        }
    }
    return max_actions;
}

static int rule_rejects_candidate(const ModularRule *rule, uint64_t state, size_t action)
{
    uint64_t candidate = state + (uint64_t)action;
    switch (rule->kind) {
    case RULE_TUPLE_DIVISIBLE: {
        uint64_t member = candidate + rule->tuple_offset;
        return member > rule->prime && (member % rule->prime) == 0u;
    }
    case RULE_KFREE_DIVISIBLE:
        return (candidate % rule->modulus) == 0u;
    case RULE_RECURRENCE_ZERO:
        return (candidate % rule->modulus) == rule->residue;
    default:
        return 0;
    }
}

static void init_rule(ModularRule *rule, ModularTask task, ModularRuleKind kind)
{
    memset(rule, 0, sizeof(*rule));
    rule->task = task;
    rule->kind = kind;
}

static int add_rule(ModularRule *rules, size_t capacity, size_t *count, const ModularRule *rule)
{
    if (*count >= capacity) {
        return 0;
    }
    rules[*count] = *rule;
    rules[*count].rule_index = *count;
    ++(*count);
    return 1;
}

static size_t enumerate_prime_constellation_rules(ModularRule *rules, size_t capacity)
{
    size_t count = 0u;
    for (size_t i = 0u; i < sizeof(PRIME_RULE_PRIMES) / sizeof(PRIME_RULE_PRIMES[0]); ++i) {
        for (uint32_t offset_index = 0u; offset_index < 2u; ++offset_index) {
            ModularRule rule;
            uint32_t offset = offset_index == 0u ? 0u : 2u;
            init_rule(&rule, TASK_PRIME_CONSTELLATION, RULE_TUPLE_DIVISIBLE);
            rule.prime = PRIME_RULE_PRIMES[i];
            rule.modulus = rule.prime;
            rule.tuple_offset = offset;
            rule.certified = 1;
            snprintf(rule.text,
                     sizeof(rule.text),
                     "reject twin-prime shift when n+d+%u == 0 mod %u",
                     offset,
                     rule.prime);
            if (!add_rule(rules, capacity, &count, &rule)) {
                return count;
            }
        }
    }
    return count;
}

static size_t enumerate_kfree_rules(ModularRule *rules, size_t capacity)
{
    size_t count = 0u;
    for (size_t i = 0u; i < sizeof(KFREE_RULE_PRIMES) / sizeof(KFREE_RULE_PRIMES[0]); ++i) {
        for (uint32_t power = 2u; power <= 3u; ++power) {
            ModularRule rule;
            init_rule(&rule, TASK_KFREE, RULE_KFREE_DIVISIBLE);
            rule.prime = KFREE_RULE_PRIMES[i];
            rule.power = power;
            rule.modulus = pow_u32(rule.prime, power);
            rule.certified = 1;
            snprintf(rule.text,
                     sizeof(rule.text),
                     "reject value when value == 0 mod %u^%u",
                     rule.prime,
                     power);
            if (!add_rule(rules, capacity, &count, &rule)) {
                return count;
            }
        }
    }
    return count;
}

static size_t enumerate_recurrence_rules(ModularRule *rules, size_t capacity)
{
    size_t count = 0u;
    for (size_t i = 0u; i < sizeof(RECURRENCE_RULE_PRIMES) / sizeof(RECURRENCE_RULE_PRIMES[0]); ++i) {
        uint32_t prime = RECURRENCE_RULE_PRIMES[i];
        uint32_t period = pisano_period_u32(prime);
        if (period == 0u) {
            continue;
        }
        for (uint32_t sequence = SEQUENCE_FIBONACCI; sequence <= SEQUENCE_LUCAS; ++sequence) {
            for (uint32_t residue = 0u; residue < period; ++residue) {
                if (sequence_mod_u32((RecurrenceSequence)sequence, residue, prime) != 0u) {
                    continue;
                }
                ModularRule rule;
                init_rule(&rule, TASK_FIBONACCI_LUCAS, RULE_RECURRENCE_ZERO);
                rule.sequence = (RecurrenceSequence)sequence;
                rule.prime = prime;
                rule.modulus = period;
                rule.residue = residue;
                rule.certified = 1;
                snprintf(rule.text,
                         sizeof(rule.text),
                         "reject %s index when i == %u mod period_%u_%u",
                         sequence_name(rule.sequence),
                         residue,
                         prime,
                         period);
                if (!add_rule(rules, capacity, &count, &rule)) {
                    return count;
                }
            }
        }
    }
    return count;
}

static size_t enumerate_rules(ModularTask task, ModularRule *rules, size_t capacity)
{
    switch (task) {
    case TASK_PRIME_CONSTELLATION:
        return enumerate_prime_constellation_rules(rules, capacity);
    case TASK_KFREE:
        return enumerate_kfree_rules(rules, capacity);
    case TASK_FIBONACCI_LUCAS:
        return enumerate_recurrence_rules(rules, capacity);
    default:
        return 0u;
    }
}

static void audit_rule_on_split(const ModularRule *rule,
                                ModularTask task,
                                size_t max_states,
                                size_t max_actions,
                                ModularSplit split,
                                size_t *out_support,
                                size_t *out_false_rejections)
{
    size_t support = 0u;
    size_t false_rejections = 0u;
    for (size_t i = 0u; i < max_states; ++i) {
        uint64_t state = state_base(task, i, split);
        size_t target = true_action_for_state(task, state, max_actions);
        for (size_t action = 0u; action < max_actions; ++action) {
            if (!rule_rejects_candidate(rule, state, action)) {
                continue;
            }
            ++support;
            if (target < max_actions && action == target) {
                ++false_rejections;
            }
        }
    }
    *out_support = support;
    *out_false_rejections = false_rejections;
}

static void score_rule_on_split(ModularRule *rule,
                                ModularTask task,
                                size_t max_states,
                                size_t max_actions,
                                ModularSplit split)
{
    if (split == SPLIT_VALIDATION) {
        audit_rule_on_split(rule,
                            task,
                            max_states,
                            max_actions,
                            split,
                            &rule->validation_support,
                            &rule->validation_false_rejections);
    } else {
        audit_rule_on_split(rule,
                            task,
                            max_states,
                            max_actions,
                            split,
                            &rule->train_support,
                            &rule->train_false_rejections);
    }
}

static void clear_rule_selection(ModularRule *rules, size_t rule_count)
{
    for (size_t i = 0u; i < rule_count; ++i) {
        rules[i].accepted = 0;
        rules[i].accepted_rank = 0u;
        rules[i].train_marginal = 0u;
        rules[i].validation_marginal = 0u;
    }
}

static int rule_is_safe_candidate(const ModularRule *rule)
{
    return rule->certified &&
           rule->train_false_rejections == 0u &&
           rule->validation_false_rejections == 0u &&
           (rule->train_support != 0u || rule->validation_support != 0u);
}

static size_t split_marginal_utility(const ModularRule *rule,
                                     ModularTask task,
                                     size_t max_states,
                                     size_t max_actions,
                                     ModularSplit split,
                                     const unsigned char *covered)
{
    size_t marginal = 0u;
    for (size_t i = 0u; i < max_states; ++i) {
        uint64_t state = state_base(task, i, split);
        for (size_t action = 0u; action < max_actions; ++action) {
            size_t event = i * max_actions + action;
            if (!covered[event] && rule_rejects_candidate(rule, state, action)) {
                ++marginal;
            }
        }
    }
    return marginal;
}

static void mark_split_coverage(const ModularRule *rule,
                                ModularTask task,
                                size_t max_states,
                                size_t max_actions,
                                ModularSplit split,
                                unsigned char *covered)
{
    for (size_t i = 0u; i < max_states; ++i) {
        uint64_t state = state_base(task, i, split);
        for (size_t action = 0u; action < max_actions; ++action) {
            size_t event = i * max_actions + action;
            if (rule_rejects_candidate(rule, state, action)) {
                covered[event] = 1u;
            }
        }
    }
}

static uint64_t mix_u64(uint64_t value)
{
    value ^= value >> 33u;
    value *= UINT64_C(0xff51afd7ed558ccd);
    value ^= value >> 33u;
    value *= UINT64_C(0xc4ceb9fe1a85ec53);
    value ^= value >> 33u;
    return value;
}

static void encode_learned_features(const ModularRule *rule, double *features)
{
    double modulus = rule->modulus == 0u ? 1.0 : (double)rule->modulus;
    features[0] = 1.0;
    features[1] = 1.0 / modulus;
    features[2] = rule->residue == 0u ? 1.0 : (double)rule->residue / modulus;
    features[3] = rule->power == 0u ? 0.0 : (double)rule->power / 3.0;
    features[4] = rule->tuple_offset == 0u ? 0.0 : (double)rule->tuple_offset / 2.0;
    features[5] = rule->sequence == SEQUENCE_FIBONACCI ? 1.0 : 0.0;
    features[6] = rule->sequence == SEQUENCE_LUCAS ? 1.0 : 0.0;
    features[7] = rule->kind == RULE_RECURRENCE_ZERO ? 1.0 :
                  (rule->kind == RULE_KFREE_DIVISIBLE ? 0.5 : 0.0);
}

static LearnedProposer train_learned_proposer(const ModularRule *rules,
                                              size_t rule_count,
                                              size_t max_states,
                                              size_t max_actions)
{
    LearnedProposer proposer;
    memset(&proposer, 0, sizeof(proposer));
    double events = (double)(max_states == 0u ? 1u : max_states) *
                    (double)(max_actions == 0u ? 1u : max_actions);
    const double learning_rate = 0.18;
    const double l2 = 0.0005;
    for (size_t epoch = 0u; epoch < 384u; ++epoch) {
        for (size_t i = 0u; i < rule_count; ++i) {
            double features[LEARNED_FEATURE_COUNT];
            double prediction = 0.0;
            double target = (double)rules[i].train_support / events;
            encode_learned_features(&rules[i], features);
            for (size_t j = 0u; j < LEARNED_FEATURE_COUNT; ++j) {
                prediction += proposer.weights[j] * features[j];
            }
            double error = prediction - target;
            for (size_t j = 0u; j < LEARNED_FEATURE_COUNT; ++j) {
                proposer.weights[j] -= learning_rate *
                                       (error * features[j] + l2 * proposer.weights[j]);
            }
        }
    }

    double loss = 0.0;
    for (size_t i = 0u; i < rule_count; ++i) {
        double features[LEARNED_FEATURE_COUNT];
        double prediction = 0.0;
        double target = (double)rules[i].train_support / events;
        encode_learned_features(&rules[i], features);
        for (size_t j = 0u; j < LEARNED_FEATURE_COUNT; ++j) {
            prediction += proposer.weights[j] * features[j];
        }
        double error = prediction - target;
        loss += error * error;
    }
    proposer.loss = rule_count == 0u ? 0.0 : loss / (double)rule_count;
    return proposer;
}

static double structural_score(const ModularRule *rule)
{
    double modulus = rule->modulus == 0u ? 1.0 : (double)rule->modulus;
    double score = 1.0 / modulus;
    if (rule->kind == RULE_KFREE_DIVISIBLE && rule->power == 2u) {
        score += 0.01;
    }
    if (rule->kind == RULE_RECURRENCE_ZERO) {
        score += rule->sequence == SEQUENCE_FIBONACCI ? 0.008 : 0.004;
    }
    return score;
}

static double learned_score(const LearnedProposer *proposer, const ModularRule *rule)
{
    double features[LEARNED_FEATURE_COUNT];
    double score = 0.0;
    encode_learned_features(rule, features);
    for (size_t i = 0u; i < LEARNED_FEATURE_COUNT; ++i) {
        score += proposer->weights[i] * features[i];
    }
    return score;
}

static double order_score(ModularSelectionStrategy strategy,
                          const ModularRule *rule,
                          size_t index,
                          ModularTask task,
                          uint64_t seed,
                          const LearnedProposer *proposer)
{
    switch (strategy) {
    case STRATEGY_GRAMMAR_ORDER:
        return -(double)index;
    case STRATEGY_RANDOM: {
        uint64_t key = mix_u64(seed ^ ((uint64_t)task << 48u) ^ (uint64_t)index);
        return (double)(key >> 11u) / (double)(UINT64_C(1) << 53u);
    }
    case STRATEGY_STRUCTURAL:
        return structural_score(rule);
    case STRATEGY_LEARNED_STATIC:
        return learned_score(proposer, rule);
    case STRATEGY_TRAIN_MARGINAL_ORACLE:
        return (double)rule->train_support;
    case STRATEGY_VALIDATION_MARGINAL_ORACLE:
        return (double)rule->validation_support;
    default:
        return 0.0;
    }
}

static void build_ranked_order(const ModularRule *rules,
                               size_t rule_count,
                               ModularSelectionStrategy strategy,
                               ModularTask task,
                               uint64_t seed,
                               const LearnedProposer *proposer,
                               size_t *order)
{
    for (size_t i = 0u; i < rule_count; ++i) {
        order[i] = i;
    }
    for (size_t i = 0u; i < rule_count; ++i) {
        size_t best = i;
        double best_score = order_score(strategy,
                                        &rules[order[i]],
                                        order[i],
                                        task,
                                        seed,
                                        proposer);
        for (size_t j = i + 1u; j < rule_count; ++j) {
            double score = order_score(strategy,
                                       &rules[order[j]],
                                       order[j],
                                       task,
                                       seed,
                                       proposer);
            if (score > best_score) {
                best = j;
                best_score = score;
            }
        }
        if (best != i) {
            size_t tmp = order[i];
            order[i] = order[best];
            order[best] = tmp;
        }
    }
}

static void set_rule_marginal(ModularRule *rule, ModularSplit split, size_t marginal)
{
    if (split == SPLIT_VALIDATION) {
        rule->validation_marginal = marginal;
    } else {
        rule->train_marginal = marginal;
    }
}

static size_t select_rules_by_fixed_order(ModularTask task,
                                          ModularRule *rules,
                                          size_t rule_count,
                                          size_t max_states,
                                          size_t max_actions,
                                          size_t max_rules,
                                          ModularSplit marginal_split,
                                          const size_t *order,
                                          size_t *accepted)
{
    size_t event_count = max_states * max_actions;
    unsigned char *covered = (unsigned char *)calloc(event_count == 0u ? 1u : event_count, 1u);
    size_t accepted_count = 0u;
    if (covered == NULL) {
        return 0u;
    }
    for (size_t rank = 0u; rank < rule_count && accepted_count < max_rules; ++rank) {
        size_t r = order[rank];
        if (!rule_is_safe_candidate(&rules[r])) {
            continue;
        }
        size_t marginal = split_marginal_utility(&rules[r],
                                                 task,
                                                 max_states,
                                                 max_actions,
                                                 marginal_split,
                                                 covered);
        if (marginal == 0u) {
            continue;
        }
        set_rule_marginal(&rules[r], marginal_split, marginal);
        rules[r].accepted = 1;
        rules[r].accepted_rank = accepted_count + 1u;
        accepted[accepted_count++] = r;
        mark_split_coverage(&rules[r], task, max_states, max_actions, marginal_split, covered);
    }
    free(covered);
    return accepted_count;
}

static size_t select_rules_greedy(ModularTask task,
                                  ModularRule *rules,
                                  size_t rule_count,
                                  size_t max_states,
                                  size_t max_actions,
                                  size_t max_rules,
                                  ModularSplit marginal_split,
                                  size_t *accepted)
{
    size_t event_count = max_states * max_actions;
    unsigned char *covered = (unsigned char *)calloc(event_count == 0u ? 1u : event_count, 1u);
    unsigned char *selected = (unsigned char *)calloc(rule_count == 0u ? 1u : rule_count, 1u);
    size_t accepted_count = 0u;
    if (covered == NULL || selected == NULL) {
        free(covered);
        free(selected);
        return 0u;
    }

    while (accepted_count < max_rules) {
        size_t best = rule_count;
        size_t best_marginal = 0u;
        for (size_t r = 0u; r < rule_count; ++r) {
            if (selected[r] || !rule_is_safe_candidate(&rules[r])) {
                continue;
            }
            size_t marginal = split_marginal_utility(&rules[r],
                                                     task,
                                                     max_states,
                                                     max_actions,
                                                     marginal_split,
                                                     covered);
            if (marginal > best_marginal) {
                best_marginal = marginal;
                best = r;
            }
        }
        if (best == rule_count || best_marginal == 0u) {
            break;
        }
        selected[best] = 1u;
        set_rule_marginal(&rules[best], marginal_split, best_marginal);
        rules[best].accepted = 1;
        rules[best].accepted_rank = accepted_count + 1u;
        accepted[accepted_count++] = best;
        mark_split_coverage(&rules[best], task, max_states, max_actions, marginal_split, covered);
    }

    free(covered);
    free(selected);
    return accepted_count;
}

static size_t select_rules(ModularTask task,
                           ModularRule *rules,
                           size_t rule_count,
                           size_t max_states,
                           size_t max_actions,
                           size_t max_rules,
                           ModularSelectionStrategy strategy,
                           uint64_t seed,
                           const LearnedProposer *proposer,
                           size_t *accepted)
{
    clear_rule_selection(rules, rule_count);
    if (strategy == STRATEGY_TRAIN_MARGINAL_ORACLE ||
        strategy == STRATEGY_VALIDATION_MARGINAL_ORACLE) {
        return select_rules_greedy(task,
                                   rules,
                                   rule_count,
                                   max_states,
                                   max_actions,
                                   max_rules,
                                   strategy == STRATEGY_VALIDATION_MARGINAL_ORACLE
                                       ? SPLIT_VALIDATION
                                       : SPLIT_TRAIN,
                                   accepted);
    }

    size_t *order = (size_t *)malloc((rule_count == 0u ? 1u : rule_count) * sizeof(*order));
    size_t accepted_count = 0u;
    if (order == NULL) {
        return 0u;
    }
    build_ranked_order(rules, rule_count, strategy, task, seed, proposer, order);
    accepted_count = select_rules_by_fixed_order(task,
                                                 rules,
                                                 rule_count,
                                                 max_states,
                                                 max_actions,
                                                 max_rules,
                                                 SPLIT_TRAIN,
                                                 order,
                                                 accepted);
    free(order);
    return accepted_count;
}

static int accepted_rules_reject(const ModularRule *rules,
                                 const size_t *accepted,
                                 size_t accepted_count,
                                 uint64_t state,
                                 size_t action)
{
    for (size_t i = 0u; i < accepted_count; ++i) {
        if (rule_rejects_candidate(&rules[accepted[i]], state, action)) {
            return 1;
        }
    }
    return 0;
}

static ModularGateSummary evaluate_gate(ModularTask task,
                                        ModularRule *rules,
                                        size_t rule_count,
                                        const size_t *accepted,
                                        size_t accepted_count,
                                        size_t max_states,
                                        size_t max_actions)
{
    ModularGateSummary summary;
    memset(&summary, 0, sizeof(summary));
    summary.proposed_rules = rule_count;
    summary.accepted_rules = accepted_count;
    summary.states = max_states;
    for (size_t i = 0u; i < rule_count; ++i) {
        if (rules[i].certified) {
            ++summary.certified_rules;
        }
    }

    for (size_t i = 0u; i < max_states; ++i) {
        uint64_t state = state_base(task, i, SPLIT_TEST);
        size_t target = true_action_for_state(task, state, max_actions);
        size_t survivors = 0u;
        if (target < max_actions) {
            ++summary.states_with_target;
        }
        for (size_t action = 0u; action < max_actions; ++action) {
            int rejected = accepted_rules_reject(rules, accepted, accepted_count, state, action);
            if (rejected) {
                ++summary.rejected_candidates;
                if (target < max_actions && action == target) {
                    ++summary.false_rejections;
                }
            } else {
                ++survivors;
            }
        }
        summary.baseline_candidates += max_actions;
        summary.gated_candidates += survivors;
    }

    return summary;
}

static const char *rule_verifier_result(const ModularRule *rule)
{
    if (rule->accepted) {
        return "accepted";
    }
    if (rule->train_false_rejections != 0u ||
        rule->validation_false_rejections != 0u) {
        return "unsafe";
    }
    if (rule->train_support == 0u && rule->validation_support == 0u) {
        return "no_support";
    }
    return "verified_not_selected";
}

static void write_rule_trace(FILE *trace,
                             const ModularRule *rule,
                             ModularTask task,
                             ModularSelectionStrategy strategy,
                             uint64_t seed,
                             const LearnedProposer *proposer)
{
    if (trace == NULL) {
        return;
    }
    fprintf(trace,
            "{\"schema\":\"crlp.modular_gate.rule_trace.v1\","
            "\"task\":\"%s\","
            "\"selection_strategy\":\"%s\","
            "\"rule_kind\":\"%s\","
            "\"accepted\":%s,"
            "\"rank\":%zu,"
            "\"verifier_result\":\"%s\","
            "\"rule_text\":\"%s\","
            "\"certificate\":\"%s\","
            "\"prime\":%u,"
            "\"power\":%u,"
            "\"modulus\":%u,"
            "\"residue\":%u,"
            "\"tuple_offset\":%u,"
            "\"sequence\":\"%s\","
            "\"model_score\":%.12g,"
            "\"train_support\":%zu,"
            "\"train_marginal_utility\":%zu,"
            "\"train_false_rejections\":%zu,"
            "\"validation_support\":%zu,"
            "\"validation_marginal_utility\":%zu,"
            "\"validation_false_rejections\":%zu}\n",
            task_name(task),
            strategy_name(strategy),
            rule_kind_name(rule->kind),
            rule->accepted ? "true" : "false",
            rule->accepted_rank,
            rule_verifier_result(rule),
            rule->text,
            certificate_name(task),
            rule->prime,
            rule->power,
            rule->modulus,
            rule->residue,
            rule->tuple_offset,
            sequence_name(rule->sequence),
            order_score(strategy, rule, rule->rule_index, task, seed, proposer),
            rule->train_support,
            rule->train_marginal,
            rule->train_false_rejections,
            rule->validation_support,
            rule->validation_marginal,
            rule->validation_false_rejections);
}

static double summary_saved_pct(const ModularGateSummary *summary)
{
    return summary->baseline_candidates == 0u
               ? 0.0
               : 100.0 * (double)summary->rejected_candidates /
                     (double)summary->baseline_candidates;
}

static int run_task_strategy(ModularTask task,
                             const ModularGateConfig *config,
                             const ModularRule *base_rules,
                             size_t rule_count,
                             ModularSelectionStrategy strategy,
                             const LearnedProposer *proposer,
                             FILE *trace)
{
    ModularRule rules[256];
    size_t accepted[256];
    if (rule_count > sizeof(rules) / sizeof(rules[0])) {
        return 0;
    }
    memcpy(rules, base_rules, rule_count * sizeof(rules[0]));
    size_t accepted_count = select_rules(task,
                                         rules,
                                         rule_count,
                                         config->max_states,
                                         config->max_actions,
                                         config->max_rules,
                                         strategy,
                                         config->selection_seed,
                                         proposer,
                                         accepted);
    ModularGateSummary summary = evaluate_gate(task,
                                               rules,
                                               rule_count,
                                               accepted,
                                               accepted_count,
                                               config->max_states,
                                               config->max_actions);
    double saved_pct = summary_saved_pct(&summary);
    printf("{\"schema\":\"crlp.modular_gate.summary.v1\","
           "\"task\":\"%s\","
           "\"selection_strategy\":\"%s\","
           "\"candidate_count\":%zu,"
           "\"train_states\":%zu,"
           "\"validation_states\":%zu,"
           "\"test_states\":%zu,"
           "\"test_states_with_target\":%zu,"
           "\"max_rules\":%zu,"
           "\"selection_seed\":%llu,"
           "\"proposed_rules\":%zu,"
           "\"certified_rules\":%zu,"
           "\"accepted_rules\":%zu,"
           "\"proposer_feature_set\":\"%s\","
           "\"proposer_target\":\"%s\","
           "\"proposer_loss\":%.12g,"
           "\"false_rejections\":%zu,"
           "\"baseline_candidates\":%zu,"
           "\"gated_candidates\":%zu,"
           "\"rejected_candidates\":%zu,"
           "\"saved_pct\":%.8f,"
           "\"certificate\":\"%s\"}\n",
           task_name(task),
           strategy_name(strategy),
           config->max_actions,
           config->max_states,
           config->max_states,
           summary.states,
           summary.states_with_target,
           config->max_rules,
           (unsigned long long)config->selection_seed,
           summary.proposed_rules,
           summary.certified_rules,
           summary.accepted_rules,
           strategy_feature_set(strategy),
           strategy_target(strategy),
           strategy == STRATEGY_LEARNED_STATIC ? proposer->loss : 0.0,
           summary.false_rejections,
           summary.baseline_candidates,
           summary.gated_candidates,
           summary.rejected_candidates,
           saved_pct,
           certificate_name(task));

    printf("[certified modular gate] task=%s strategy=%s budget=%zu accepted=%zu/%zu saved=%.2f%% false_rejections=%zu\n",
           task_name(task),
           strategy_name(strategy),
           config->max_rules,
           summary.accepted_rules,
           summary.certified_rules,
           saved_pct,
           summary.false_rejections);

    for (size_t i = 0u; i < rule_count; ++i) {
        write_rule_trace(trace,
                         &rules[i],
                         task,
                         strategy,
                         config->selection_seed,
                         proposer);
    }

    return summary.false_rejections == 0u && summary.accepted_rules > 0u;
}

static int run_task(ModularTask task, const ModularGateConfig *config, FILE *trace)
{
    enum { MAX_RULES = 256 };
    ModularRule rules[MAX_RULES];
    size_t rule_count = enumerate_rules(task, rules, MAX_RULES);
    for (size_t i = 0u; i < rule_count; ++i) {
        score_rule_on_split(&rules[i], task, config->max_states, config->max_actions, SPLIT_TRAIN);
        score_rule_on_split(&rules[i], task, config->max_states, config->max_actions, SPLIT_VALIDATION);
    }
    LearnedProposer proposer = train_learned_proposer(rules,
                                                      rule_count,
                                                      config->max_states,
                                                      config->max_actions);
    int ok = 1;
    for (ModularSelectionStrategy strategy = 0; strategy < STRATEGY_COUNT; ++strategy) {
        ok = run_task_strategy(task,
                               config,
                               rules,
                               rule_count,
                               strategy,
                               &proposer,
                               trace) && ok;
    }
    return ok;
}

static ModularGateConfig default_config(void)
{
    ModularGateConfig config;
    config.max_states = 512u;
    config.max_actions = 32u;
    config.max_rules = 16u;
    config.task = TASK_PRIME_CONSTELLATION;
    config.run_all = 1;
    config.selection_seed = UINT64_C(20240511);
    config.trace_path = NULL;
    return config;
}

static int parse_task(const char *name, ModularGateConfig *config)
{
    if (strcmp(name, "all") == 0) {
        config->run_all = 1;
        return 1;
    }
    config->run_all = 0;
    if (strcmp(name, "prime-constellation") == 0 || strcmp(name, "prime_constellation") == 0) {
        config->task = TASK_PRIME_CONSTELLATION;
        return 1;
    }
    if (strcmp(name, "kfree") == 0 || strcmp(name, "squarefree") == 0 ||
        strcmp(name, "kfree_squarefree") == 0) {
        config->task = TASK_KFREE;
        return 1;
    }
    if (strcmp(name, "fibonacci-lucas") == 0 || strcmp(name, "fibonacci_lucas") == 0 ||
        strcmp(name, "recurrence") == 0) {
        config->task = TASK_FIBONACCI_LUCAS;
        return 1;
    }
    return 0;
}

static int parse_size_arg(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed == 0ul) {
        return 0;
    }
    *value = (size_t)parsed;
    return 1;
}

static void print_usage(FILE *stream)
{
    fputs("usage: modular_gate_search [--task all|prime-constellation|kfree|fibonacci-lucas] "
          "[--max-states n] [--max-actions n] [--max-rules n] "
          "[--selection-seed n] [--trace path]\n",
          stream);
}

static int parse_config(int argc, char **argv, ModularGateConfig *config)
{
    *config = default_config();
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(stdout);
            exit(0);
        }
        if (strcmp(argv[i], "--task") == 0 && i + 1 < argc) {
            if (!parse_task(argv[++i], config)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-states") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_states)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-actions") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_actions)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-rules") == 0 && i + 1 < argc) {
            if (!parse_size_arg(argv[++i], &config->max_rules)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--selection-seed") == 0 && i + 1 < argc) {
            size_t seed = 0u;
            if (!parse_size_arg(argv[++i], &seed)) {
                return 0;
            }
            config->selection_seed = (uint64_t)seed;
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else {
            return 0;
        }
    }
    if (config->max_actions > 128u || config->max_rules > 256u) {
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    ModularGateConfig config;
    FILE *trace = NULL;
    int ok = 1;
    if (!parse_config(argc, argv, &config)) {
        print_usage(stderr);
        return 2;
    }
    if (config.trace_path != NULL) {
        trace = fopen(config.trace_path, "w");
        if (trace == NULL) {
            fprintf(stderr, "failed to open trace: %s\n", config.trace_path);
            return 1;
        }
    }

    if (config.run_all) {
        for (ModularTask task = 0; task < TASK_COUNT; ++task) {
            ok = run_task(task, &config, trace) && ok;
        }
    } else {
        ok = run_task(config.task, &config, trace);
    }

    if (trace != NULL) {
        fclose(trace);
    }
    return ok ? 0 : 1;
}
