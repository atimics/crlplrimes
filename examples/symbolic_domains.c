#include "nn.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FORMAL_FEATURE_COUNT = 16,
    CLAIM_STATE_FEATURES = 6,
    CLAIM_ACTION_COUNT = 4,
    ARC_INPUT_W = 7,
    ARC_INPUT_H = 3,
    ARC_INPUT_CELLS = ARC_INPUT_W * ARC_INPUT_H,
    ARC_OUTPUT_W = 3,
    ARC_OUTPUT_H = 3,
    ARC_OUTPUT_CELLS = ARC_OUTPUT_W * ARC_OUTPUT_H,
    ARC_PAIR_COUNT = 3,
};

typedef enum {
    PRIME_RULE_CANDIDATE_MOD = 0,
    PRIME_RULE_CONDITIONAL_GAP_MOD = 1,
} PrimeRuleTemplate;

typedef enum {
    GRID_RULE_UP_AT_TOP = 0,
    GRID_RULE_RIGHT_AT_EDGE = 1,
    GRID_RULE_DOWN_AT_BOTTOM = 2,
    GRID_RULE_LEFT_AT_EDGE = 3,
    GRID_RULE_BLOCKED_NEXT = 4,
    GRID_RULE_DEAD_END_NEXT = 5,
    GRID_RULE_COMPONENT_TRAP_NEXT = 6,
    GRID_RULE_CLEAR_AXIS_DOMINATED_NEXT = 7,
    GRID_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT = 8,
    GRID_RULE_DISTANCE_DOMINATED_NEXT = 9,
} GridRuleTemplate;

typedef enum {
    CLAIM_RULE_REJECT_TRUST_LOW_RELIABILITY = 0,
    CLAIM_RULE_REJECT_TRUST_HIGH_CONTRADICTION = 1,
    CLAIM_RULE_REJECT_IGNORE_HIGH_EVIDENCE = 2,
    CLAIM_RULE_REJECT_CAUTIOUS_LOW_EVIDENCE = 3,
    CLAIM_RULE_REJECT_TRUST_IF_REQUEST_BETTER = 4,
    CLAIM_RULE_REJECT_REQUEST_IF_TRUST_BETTER = 5,
    CLAIM_RULE_REJECT_TRUST_IF_REQUEST_PROXY_BETTER = 6,
    CLAIM_RULE_REJECT_REQUEST_IF_TRUST_PROXY_BETTER = 7,
} ClaimRuleTemplate;

typedef enum {
    ARC_RULE_COPY_LEFT = 0,
    ARC_RULE_COPY_RIGHT = 1,
    ARC_RULE_INTERSECTION = 2,
    ARC_RULE_UNION = 3,
    ARC_RULE_OBJECT_SELECT = 4,
    ARC_RULE_COLOR_MAP_LEFT = 5,
    ARC_RULE_LAYER_OVERLAP = 6,
    ARC_RULE_LAYER_UNION = 7,
    ARC_RULE_LAYER_XOR = 8,
    ARC_RULE_LAYER_LEFT_ONLY = 9,
    ARC_RULE_LAYER_RIGHT_ONLY = 10,
    ARC_RULE_COUNT = 11,
} ArcRuleTemplate;

typedef struct {
    double state[CLAIM_STATE_FEATURES];
    double utility[CLAIM_ACTION_COUNT];
    double risk_proxy[CLAIM_ACTION_COUNT];
    size_t target;
} ClaimStateLite;

static const NNFeatureSchema FORMAL_RULE_FEATURE_SCHEMA[FORMAL_FEATURE_COUNT] = {
    {"domain_tag", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"rule_kind", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"template_id", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_0", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_1", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_2", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_3", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_4", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_5", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_6", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_7", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_8", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_9", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_10", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"param_11", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"padding", NN_FEATURE_STRUCTURAL_PRIOR, 1},
};

static const uint8_t ARC_TRAIN_INPUTS[ARC_PAIR_COUNT][ARC_INPUT_CELLS] = {
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

static const uint8_t ARC_TRAIN_OUTPUTS[ARC_PAIR_COUNT][ARC_OUTPUT_CELLS] = {
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

static const uint8_t ARC_TEST_INPUT[ARC_INPUT_CELLS] = {
    1, 0, 1, 5, 1, 0, 1,
    0, 1, 0, 5, 1, 0, 1,
    1, 0, 1, 5, 0, 1, 0,
};

static uint32_t formal_gcd_u32(uint32_t a, uint32_t b)
{
    while (b != 0u) {
        uint32_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static uint32_t claim_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double claim_rng_unit(uint32_t *state)
{
    return (double)claim_rng_next(state) / (double)UINT32_MAX;
}

static void compute_claim_lite_utilities(const double features[CLAIM_STATE_FEATURES],
                                         double utility[CLAIM_ACTION_COUNT])
{
    double reliability = features[0];
    double evidence = features[1];
    double contradiction = features[2];
    double age = features[3];
    double history = features[4];
    double noise = features[5];
    double truth_score = 0.45 * reliability + 0.35 * evidence + 0.20 * history -
                         0.30 * contradiction - 0.15 * noise - 0.10 * age;
    int resolves_true = truth_score >= 0.38;
    double uncertainty = fabs(truth_score - 0.38);

    utility[0] = resolves_true ? 2.0 : -2.5;
    utility[1] = resolves_true ? -0.3 : 0.3;
    utility[2] = 0.95 - 0.6 * uncertainty - 0.25 * age;
    utility[3] = resolves_true ? 1.1 : -0.2;
    if (contradiction > 0.70 || noise > 0.80) {
        utility[2] += 0.35;
        utility[0] -= 0.75;
    }
}

static void compute_claim_lite_risk_proxy(const double features[CLAIM_STATE_FEATURES],
                                          double proxy[CLAIM_ACTION_COUNT])
{
    double reliability = features[0];
    double evidence = features[1];
    double contradiction = features[2];
    double age = features[3];
    double history = features[4];
    double noise = features[5];
    double truth_score = 0.45 * reliability + 0.35 * evidence + 0.20 * history -
                         0.30 * contradiction - 0.15 * noise - 0.10 * age;
    double high_risk_excess = fmax(0.0, contradiction - 0.70) +
                              fmax(0.0, noise - 0.80);
    memset(proxy, 0, CLAIM_ACTION_COUNT * sizeof(*proxy));
    proxy[0] = truth_score - 0.45 * high_risk_excess - 0.04 * age;
    proxy[2] = 0.38 - truth_score + 0.35 * high_risk_excess +
               0.08 * age + 0.06 * contradiction + 0.03 * noise;
}

static ClaimStateLite generate_claim_lite(uint64_t seed)
{
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ UINT64_C(0x9e3779b9));
    ClaimStateLite state;
    for (size_t i = 0; i < CLAIM_STATE_FEATURES; ++i) {
        state.state[i] = claim_rng_unit(&rng);
    }
    compute_claim_lite_utilities(state.state, state.utility);
    compute_claim_lite_risk_proxy(state.state, state.risk_proxy);
    state.target = 0;
    for (size_t action = 1; action < CLAIM_ACTION_COUNT; ++action) {
        if (state.utility[action] > state.utility[state.target]) {
            state.target = action;
        }
    }
    return state;
}

static void encode_rule_features(const NNRule *rule,
                                 size_t domain_id,
                                 nn_real *features,
                                 size_t capacity,
                                 size_t *feature_count_out)
{
    memset(features, 0, capacity * sizeof(*features));
    if (capacity < FORMAL_FEATURE_COUNT) {
        if (feature_count_out != NULL) {
            *feature_count_out = 0;
        }
        return;
    }
    features[0] = (nn_real)((double)domain_id / 4.0);
    features[1] = (nn_real)((double)rule->kind / 4.0);
    features[2] = (nn_real)((double)rule->template_id / 16.0);
    for (size_t i = 0; i < rule->param_count && i + 3 < FORMAL_FEATURE_COUNT; ++i) {
        features[i + 3] = (nn_real)rule->params[i].value;
    }
    if (feature_count_out != NULL) {
        *feature_count_out = FORMAL_FEATURE_COUNT;
    }
}

static int encode_prime_rule(const NNSymbolicDomain *domain,
                             const NNRule *rule,
                             nn_real *features,
                             size_t capacity,
                             size_t *feature_count_out)
{
    (void)domain;
    encode_rule_features(rule, 0, features, capacity, feature_count_out);
    return capacity >= FORMAL_FEATURE_COUNT;
}

static int encode_grid_rule(const NNSymbolicDomain *domain,
                            const NNRule *rule,
                            nn_real *features,
                            size_t capacity,
                            size_t *feature_count_out)
{
    (void)domain;
    encode_rule_features(rule, 1, features, capacity, feature_count_out);
    return capacity >= FORMAL_FEATURE_COUNT;
}

static int encode_claim_rule(const NNSymbolicDomain *domain,
                             const NNRule *rule,
                             nn_real *features,
                             size_t capacity,
                             size_t *feature_count_out)
{
    (void)domain;
    encode_rule_features(rule, 2, features, capacity, feature_count_out);
    return capacity >= FORMAL_FEATURE_COUNT;
}

static int encode_arc_rule(const NNSymbolicDomain *domain,
                           const NNRule *rule,
                           nn_real *features,
                           size_t capacity,
                           size_t *feature_count_out)
{
    (void)domain;
    encode_rule_features(rule, 3, features, capacity, feature_count_out);
    return capacity >= FORMAL_FEATURE_COUNT;
}

static size_t enumerate_prime_rules(const NNSymbolicDomain *domain,
                                    NNRule *rules,
                                    size_t capacity)
{
    (void)domain;
    if (rules == NULL || capacity == 0) {
        return 4;
    }
    size_t count = 0;
    if (count < capacity) {
        nn_rule_init(&rules[count], "prime_gap", "reject_candidate_mod_3_0",
                     NN_RULE_KIND_REJECT_ACTION, PRIME_RULE_CANDIDATE_MOD);
        nn_rule_add_param(&rules[count], "modulus", 3.0 / 16.0, 3);
        nn_rule_add_param(&rules[count], "residue", 0.0, 0);
        ++count;
    }
    if (count < capacity) {
        nn_rule_init(&rules[count], "prime_gap", "reject_candidate_mod_5_2",
                     NN_RULE_KIND_REJECT_ACTION, PRIME_RULE_CANDIDATE_MOD);
        nn_rule_add_param(&rules[count], "modulus", 5.0 / 16.0, 5);
        nn_rule_add_param(&rules[count], "residue", 2.0 / 5.0, 2);
        ++count;
    }
    if (count < capacity) {
        nn_rule_init(&rules[count], "prime_gap", "if_p_mod_5_1_reject_gap_mod_5_4",
                     NN_RULE_KIND_REJECT_ACTION, PRIME_RULE_CONDITIONAL_GAP_MOD);
        nn_rule_add_param(&rules[count], "state_modulus", 5.0 / 16.0, 5);
        nn_rule_add_param(&rules[count], "state_residue", 1.0 / 5.0, 1);
        nn_rule_add_param(&rules[count], "filter_modulus", 5.0 / 16.0, 5);
        nn_rule_add_param(&rules[count], "filter_residue", 4.0 / 5.0, 4);
        ++count;
    }
    if (count < capacity) {
        nn_rule_init(&rules[count], "prime_gap", "if_p_mod_5_1_reject_gap_mod_5_1",
                     NN_RULE_KIND_REJECT_ACTION, PRIME_RULE_CONDITIONAL_GAP_MOD);
        nn_rule_add_param(&rules[count], "state_modulus", 5.0 / 16.0, 5);
        nn_rule_add_param(&rules[count], "state_residue", 1.0 / 5.0, 1);
        nn_rule_add_param(&rules[count], "filter_modulus", 5.0 / 16.0, 5);
        nn_rule_add_param(&rules[count], "filter_residue", 1.0 / 5.0, 1);
        ++count;
    }
    return count;
}

static int check_prime_certificate(const NNSymbolicDomain *domain,
                                   const NNRule *rule,
                                   NNCertificate *certificate)
{
    (void)domain;
    int64_t modulus = 0;
    int64_t residue = 0;
    if (rule->template_id == PRIME_RULE_CANDIDATE_MOD &&
        nn_rule_get_param_i64(rule, "modulus", &modulus) &&
        nn_rule_get_param_i64(rule, "residue", &residue)) {
        if (formal_gcd_u32((uint32_t)modulus, (uint32_t)residue) > 1u) {
            nn_certificate_accept(certificate,
                                  "prime_modular_gcd",
                                  "candidate residue shares a nontrivial divisor",
                                  1,
                                  1);
        } else {
            nn_certificate_reject(certificate,
                                  "prime_modular_gcd",
                                  "candidate residue is not structurally composite",
                                  1,
                                  1);
        }
        return 1;
    }

    int64_t state_modulus = 0;
    int64_t state_residue = 0;
    int64_t filter_modulus = 0;
    int64_t filter_residue = 0;
    if (rule->template_id == PRIME_RULE_CONDITIONAL_GAP_MOD &&
        nn_rule_get_param_i64(rule, "state_modulus", &state_modulus) &&
        nn_rule_get_param_i64(rule, "state_residue", &state_residue) &&
        nn_rule_get_param_i64(rule, "filter_modulus", &filter_modulus) &&
        nn_rule_get_param_i64(rule, "filter_residue", &filter_residue) &&
        state_modulus == filter_modulus) {
        uint32_t candidate_residue =
            (uint32_t)((state_residue + filter_residue) % filter_modulus);
        if (formal_gcd_u32((uint32_t)filter_modulus, candidate_residue) > 1u) {
            nn_certificate_accept(certificate,
                                  "prime_conditional_modular_gcd",
                                  "state plus gap residue is structurally composite",
                                  1,
                                  1);
        } else {
            nn_certificate_reject(certificate,
                                  "prime_conditional_modular_gcd",
                                  "state plus gap residue is not structurally composite",
                                  1,
                                  1);
        }
        return 1;
    }

    nn_certificate_reject(certificate, "prime_modular_gcd", "malformed prime rule", 0, 1);
    return 1;
}

static size_t enumerate_grid_rules(const NNSymbolicDomain *domain,
                                   NNRule *rules,
                                   size_t capacity)
{
    (void)domain;
    static const char *names[] = {
        "reject_up_at_top",
        "reject_right_at_edge",
        "reject_down_at_bottom",
        "reject_left_at_edge",
        "reject_blocked_next",
        "reject_local_dead_end_next",
        "reject_component_trap_next",
        "reject_clear_axis_dominated_next",
        "reject_clear_manhattan_dominated_next",
        "reject_distance_dominated_next",
    };
    if (rules == NULL || capacity == 0) {
        return sizeof(names) / sizeof(names[0]);
    }
    size_t count = 0;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && count < capacity; ++i) {
        nn_rule_init(&rules[count], "grid_path", names[i], NN_RULE_KIND_REJECT_ACTION, (uint32_t)i);
        nn_rule_add_param(&rules[count],
                          "rule_id",
                          (double)i / (double)(sizeof(names) / sizeof(names[0])),
                          (int64_t)i);
        ++count;
    }
    return count;
}

static int check_grid_certificate(const NNSymbolicDomain *domain,
                                  const NNRule *rule,
                                  NNCertificate *certificate)
{
    (void)domain;
    switch ((GridRuleTemplate)rule->template_id) {
    case GRID_RULE_UP_AT_TOP:
    case GRID_RULE_RIGHT_AT_EDGE:
    case GRID_RULE_DOWN_AT_BOTTOM:
    case GRID_RULE_LEFT_AT_EDGE:
    case GRID_RULE_BLOCKED_NEXT:
        nn_certificate_accept(certificate,
                              "grid_legality_axiom",
                              "illegal moves cannot be optimal shortest-path successors",
                              1,
                              1);
        return 1;
    case GRID_RULE_DEAD_END_NEXT:
        nn_certificate_accept(certificate,
                              "grid_local_dead_end_axiom",
                              "a non-goal cul-de-sac successor cannot be optimal",
                              1,
                              1);
        return 1;
    case GRID_RULE_COMPONENT_TRAP_NEXT:
        nn_certificate_accept(certificate,
                              "grid_component_trap_axiom",
                              "a successor that can reach the goal only by backtracking through the start cannot be optimal",
                              1,
                              1);
        return 1;
    case GRID_RULE_CLEAR_AXIS_DOMINATED_NEXT:
    case GRID_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT:
        nn_certificate_accept(certificate,
                              "grid_admissible_bound_dominance",
                              "a successor whose Manhattan lower bound exceeds another successor's clear-path upper bound cannot be optimal",
                              1,
                              1);
        return 1;
    case GRID_RULE_DISTANCE_DOMINATED_NEXT:
        nn_certificate_accept(certificate,
                              "grid_shortest_path_dominance",
                              "a successor with larger goal distance than another legal successor cannot be optimal",
                              1,
                              1);
        return 1;
    }
    nn_certificate_reject(certificate,
                          "grid_shortest_path_dominance",
                          "unsupported grid rule",
                          0,
                          1);
    return 1;
}

static size_t enumerate_claim_rules(const NNSymbolicDomain *domain,
                                    NNRule *rules,
                                    size_t capacity)
{
    (void)domain;
    static const char *names[] = {
        "reject_trust_low_reliability",
        "reject_trust_high_contradiction",
        "reject_ignore_high_evidence",
        "reject_cautious_low_evidence",
    };
    static const int actions[] = {0, 0, 1, 3};
    static const int features[] = {0, 2, 1, 1};
    static const int directions[] = {-1, 1, 1, -1};
    static const double thresholds[] = {0.20, 0.85, 0.90, 0.20};
    static const char *margin_names[] = {
        "reject_trust_if_request_margin_better",
        "reject_request_if_trust_margin_better",
        "reject_trust_if_request_proxy_better",
        "reject_request_if_trust_proxy_better",
    };
    static const int margin_actions[] = {0, 2, 0, 2};
    static const int margin_other_actions[] = {2, 0, 2, 0};
    static const int margin_proxy_kind[] = {0, 0, 1, 1};
    static const double margins[] = {0.0, 0.0, 0.05, 0.50};
    size_t threshold_count = sizeof(names) / sizeof(names[0]);
    size_t margin_count = sizeof(margin_names) / sizeof(margin_names[0]);
    if (rules == NULL || capacity == 0) {
        return threshold_count + margin_count;
    }
    size_t count = 0;
    for (size_t i = 0; i < threshold_count && count < capacity; ++i) {
        nn_rule_init(&rules[count], "delayed_claim", names[i], NN_RULE_KIND_THRESHOLD, (uint32_t)i);
        nn_rule_add_param(&rules[count], "action", (double)actions[i] / 4.0, actions[i]);
        nn_rule_add_param(&rules[count], "feature", (double)features[i] / 6.0, features[i]);
        nn_rule_add_param(&rules[count], "direction", directions[i] > 0 ? 1.0 : 0.0, directions[i]);
        nn_rule_add_param(&rules[count], "threshold", thresholds[i], (int64_t)(thresholds[i] * 1000.0));
        ++count;
    }
    for (size_t i = 0; i < margin_count && count < capacity; ++i) {
        nn_rule_init(&rules[count],
                     "delayed_claim",
                     margin_names[i],
                     NN_RULE_KIND_CUSTOM,
                     (uint32_t)(CLAIM_RULE_REJECT_TRUST_IF_REQUEST_BETTER + i));
        nn_rule_add_param(&rules[count],
                          "action",
                          (double)margin_actions[i] / 4.0,
                          margin_actions[i]);
        nn_rule_add_param(&rules[count],
                          "other_action",
                          (double)margin_other_actions[i] / 4.0,
                          margin_other_actions[i]);
        nn_rule_add_param(&rules[count], "margin", margins[i], (int64_t)(margins[i] * 1000.0));
        nn_rule_add_param(&rules[count], "proxy_kind", (double)margin_proxy_kind[i], margin_proxy_kind[i]);
        ++count;
    }
    return count;
}

static int claim_rule_rejects_state(const NNRule *rule,
                                    const ClaimStateLite *state,
                                    size_t action)
{
    int64_t reject_action = 0;
    int64_t other_action = -1;
    int64_t feature = 0;
    int64_t direction = 0;
    int64_t proxy_kind = 0;
    double threshold = 0.0;
    double margin = 0.0;
    if (!nn_rule_get_param_i64(rule, "action", &reject_action)) {
        return 0;
    }
    if (action != (size_t)reject_action) {
        return 0;
    }
    if (nn_rule_get_param_i64(rule, "other_action", &other_action) &&
        nn_rule_get_param(rule, "margin", &margin)) {
        if (other_action < 0 || other_action >= CLAIM_ACTION_COUNT ||
            other_action == reject_action) {
            return 0;
        }
        (void)nn_rule_get_param_i64(rule, "proxy_kind", &proxy_kind);
        const double *scores = proxy_kind == 1 ? state->risk_proxy : state->utility;
        return scores[(size_t)other_action] > scores[(size_t)reject_action] + margin;
    }
    if (!nn_rule_get_param_i64(rule, "feature", &feature) ||
        !nn_rule_get_param_i64(rule, "direction", &direction) ||
        !nn_rule_get_param(rule, "threshold", &threshold) ||
        feature < 0 || feature >= CLAIM_STATE_FEATURES) {
        return 0;
    }
    double value = state->state[feature];
    return direction > 0 ? value >= threshold : value <= threshold;
}

static int check_claim_certificate(const NNSymbolicDomain *domain,
                                   const NNRule *rule,
                                   NNCertificate *certificate)
{
    (void)domain;
    size_t checked = 0;
    size_t support = 0;
    size_t false_rejections = 0;
    for (uint64_t seed = 1000; seed < 5000; ++seed) {
        ClaimStateLite state = generate_claim_lite(seed);
        for (size_t action = 0; action < CLAIM_ACTION_COUNT; ++action) {
            if (!claim_rule_rejects_state(rule, &state, action)) {
                continue;
            }
            ++checked;
            if (action == state.target) {
                ++false_rejections;
            } else {
                ++support;
            }
        }
    }
    if (support != 0 && false_rejections == 0) {
        nn_certificate_accept(certificate,
                              "delayed_claim_finite_generator",
                              "rule never rejects an optimal action on certified seed support",
                              checked,
                              support);
    } else {
        nn_certificate_reject(certificate,
                              "delayed_claim_finite_generator",
                              "rule has no safe support or rejects an optimal action",
                              checked,
                              false_rejections);
    }
    return 1;
}

static void arc_apply_rule(const NNRule *rule, const uint8_t *input, uint8_t *output)
{
    memset(output, 0, ARC_OUTPUT_CELLS * sizeof(*output));
    for (size_t y = 0; y < ARC_OUTPUT_H; ++y) {
        for (size_t x = 0; x < ARC_OUTPUT_W; ++x) {
            uint8_t left = input[y * ARC_INPUT_W + x];
            uint8_t right = input[y * ARC_INPUT_W + x + 4u];
            size_t out = y * ARC_OUTPUT_W + x;
            switch (rule->template_id) {
            case ARC_RULE_COPY_LEFT:
                output[out] = left == 1 ? 2 : 0;
                break;
            case ARC_RULE_COPY_RIGHT:
                output[out] = right == 1 ? 2 : 0;
                break;
            case ARC_RULE_INTERSECTION:
                output[out] = left == 1 && right == 1 ? 2 : 0;
                break;
            case ARC_RULE_UNION:
                output[out] = left == 1 || right == 1 ? 2 : 0;
                break;
            case ARC_RULE_OBJECT_SELECT:
                output[out] = left == 1 && right == 1 ? 2 : 0;
                break;
            case ARC_RULE_LAYER_OVERLAP:
                output[out] = left == 1 && right == 1 ? 2 : 0;
                break;
            case ARC_RULE_LAYER_UNION:
                output[out] = left == 1 || right == 1 ? 2 : 0;
                break;
            case ARC_RULE_LAYER_XOR:
                output[out] = (left == 1) != (right == 1) ? 2 : 0;
                break;
            case ARC_RULE_LAYER_LEFT_ONLY:
                output[out] = left == 1 && right != 1 ? 2 : 0;
                break;
            case ARC_RULE_LAYER_RIGHT_ONLY:
                output[out] = right == 1 && left != 1 ? 2 : 0;
                break;
            case ARC_RULE_COLOR_MAP_LEFT:
                output[out] = left == 1 ? 1 : 0;
                break;
            default:
                output[out] = 0;
                break;
            }
        }
    }
}

static size_t enumerate_arc_rules(const NNSymbolicDomain *domain,
                                  NNRule *rules,
                                  size_t capacity)
{
    (void)domain;
    static const char *names[] = {
        "copy_left_color_2",
        "copy_right_color_2",
        "mask_intersection_color_2",
        "mask_union_color_2",
        "object_select_overlap_color_2",
        "color_map_left_identity",
        "layer_overlap_color_2",
        "layer_union_color_2",
        "layer_xor_color_2",
        "layer_left_only_color_2",
        "layer_right_only_color_2",
    };
    if (rules == NULL || capacity == 0) {
        return sizeof(names) / sizeof(names[0]);
    }
    size_t count = 0;
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]) && count < capacity; ++i) {
        nn_rule_init(&rules[count], "arc_agi", names[i], NN_RULE_KIND_ARC_TRANSFORM, (uint32_t)i);
        nn_rule_add_param(&rules[count], "transform", (double)i / (double)ARC_RULE_COUNT, (int64_t)i);
        ++count;
    }
    return count;
}

static int arc_rule_matches_train(const NNRule *rule,
                                  size_t *checked_cells_out,
                                  size_t *mismatches_out)
{
    uint8_t produced[ARC_OUTPUT_CELLS];
    size_t checked_cells = 0;
    size_t mismatches = 0;
    for (size_t pair = 0; pair < ARC_PAIR_COUNT; ++pair) {
        arc_apply_rule(rule, ARC_TRAIN_INPUTS[pair], produced);
        for (size_t cell = 0; cell < ARC_OUTPUT_CELLS; ++cell) {
            ++checked_cells;
            if (produced[cell] != ARC_TRAIN_OUTPUTS[pair][cell]) {
                ++mismatches;
            }
        }
    }
    if (checked_cells_out != NULL) {
        *checked_cells_out = checked_cells;
    }
    if (mismatches_out != NULL) {
        *mismatches_out = mismatches;
    }
    return mismatches == 0;
}

static int check_arc_certificate(const NNSymbolicDomain *domain,
                                 const NNRule *rule,
                                 NNCertificate *certificate)
{
    (void)domain;
    size_t checked_cells = 0;
    size_t mismatches = 0;
    int matches_train = arc_rule_matches_train(rule, &checked_cells, &mismatches);
    uint8_t produced[ARC_OUTPUT_CELLS];
    arc_apply_rule(rule, ARC_TEST_INPUT, produced);
    size_t nonzero = 0;
    for (size_t cell = 0; cell < ARC_OUTPUT_CELLS; ++cell) {
        nonzero += produced[cell] != 0 ? 1u : 0u;
    }

    int ambiguous_test_output = 0;
    if (matches_train) {
        for (uint32_t template_id = 0; template_id < (uint32_t)ARC_RULE_COUNT; ++template_id) {
            NNRule other;
            nn_rule_init(&other,
                         "arc_agi",
                         "arc_uniqueness_probe",
                         NN_RULE_KIND_ARC_TRANSFORM,
                         template_id);
            if (!arc_rule_matches_train(&other, NULL, NULL)) {
                continue;
            }
            uint8_t other_output[ARC_OUTPUT_CELLS];
            arc_apply_rule(&other, ARC_TEST_INPUT, other_output);
            if (memcmp(produced, other_output, ARC_OUTPUT_CELLS * sizeof(*produced)) != 0) {
                ambiguous_test_output = 1;
                break;
            }
        }
    }

    if (matches_train && !ambiguous_test_output) {
        nn_certificate_accept(certificate,
                              "arc_train_pairs_unique_transform",
                              "transform matches all train pairs and all matching grammar rules agree on the test grid",
                              checked_cells,
                              nonzero);
    } else if (ambiguous_test_output) {
        nn_certificate_reject(certificate,
                              "arc_train_pairs_unique_transform",
                              "train pairs admit multiple grammar-consistent test outputs",
                              checked_cells,
                              1);
    } else {
        nn_certificate_reject(certificate,
                              "arc_train_pairs_unique_transform",
                              "transform does not match all train pairs",
                              checked_cells,
                              mismatches);
    }
    return 1;
}

static int rule_rejects_not_supported(const NNSymbolicDomain *domain,
                                      const NNRule *rule,
                                      const void *state,
                                      size_t action,
                                      int *rejects)
{
    (void)domain;
    (void)rule;
    (void)state;
    (void)action;
    *rejects = 0;
    return 1;
}

static int collect_domain_rules(const NNSymbolicDomain *domain,
                                NNRule *rules,
                                size_t capacity,
                                nn_real *features,
                                nn_real *targets,
                                size_t *count)
{
    if (!nn_symbolic_domain_is_valid(domain)) {
        return 0;
    }
    size_t available = nn_symbolic_domain_enumerate_rules(domain, NULL, 0);
    NNRule *local = calloc(available, sizeof(*local));
    if (local == NULL) {
        return 0;
    }
    size_t local_count = nn_symbolic_domain_enumerate_rules(domain, local, available);
    size_t certified = 0;
    for (size_t i = 0; i < local_count; ++i) {
        NNCertificate certificate;
        if (!nn_symbolic_domain_check_certificate(domain, &local[i], &certificate)) {
            free(local);
            return 0;
        }
        if (*count >= capacity) {
            free(local);
            return 0;
        }
        size_t feature_count = 0;
        if (!nn_symbolic_domain_encode_rule(domain,
                                            &local[i],
                                            &features[*count * FORMAL_FEATURE_COUNT],
                                            FORMAL_FEATURE_COUNT,
                                            &feature_count) ||
            feature_count != FORMAL_FEATURE_COUNT) {
            free(local);
            return 0;
        }
        targets[*count] =
            certificate.status == NN_CERTIFICATE_CERTIFIED ? (nn_real)1 : (nn_real)0;
        rules[*count] = local[i];
        ++*count;
        certified += certificate.status == NN_CERTIFICATE_CERTIFIED ? 1u : 0u;
    }
    NNFeatureAudit audit = nn_symbolic_domain_rule_feature_audit(domain);
    printf("{\"schema\":\"crlp.symbolic_domain.summary.v1\",\"domain\":\"%s\","
           "\"rules\":%zu,\"certified\":%zu,\"symbols\":\"%s\","
           "\"axioms\":\"%s\",\"grammar\":\"%s\","
           "\"leakage_policy\":\"%s\",\"rule_features\":%zu,"
           "\"enabled_gate_derived_features\":%zu,"
           "\"rule_feature_policy_compliant\":%s}\n",
           domain->name,
           local_count,
           certified,
           domain->symbols,
           domain->axioms,
           domain->grammar,
           nn_leakage_policy_name(domain->leakage_policy),
           audit.enabled_features,
           audit.enabled_gate_derived_features,
           audit.scorer_input_is_policy_compliant ? "true" : "false");
    free(local);
    return 1;
}

int main(void)
{
    NNSymbolicDomain domains[] = {
        {
            .name = "prime_gap",
            .description = "modular prime-gap rejection rules",
            .symbols = "p,g,n,m,r,gcd,mod",
            .axioms = "gcd(m,r)>1 and n>m implies n composite",
            .grammar = "candidate_mod, conditional_gap_mod",
            .max_rule_count = 4,
            .rule_feature_count = FORMAL_FEATURE_COUNT,
            .leakage_policy = NN_LEAKAGE_PRIORS_ALLOWED,
            .rule_feature_schema = FORMAL_RULE_FEATURE_SCHEMA,
            .rule_feature_schema_count = sizeof(FORMAL_RULE_FEATURE_SCHEMA) /
                                         sizeof(FORMAL_RULE_FEATURE_SCHEMA[0]),
            .max_action_count = 129,
            .enumerate_rules = enumerate_prime_rules,
            .encode_rule = encode_prime_rule,
            .check_certificate = check_prime_certificate,
            .rule_rejects = rule_rejects_not_supported,
        },
        {
            .name = "grid_path",
            .description = "grid legality, local-dead-end, component-trap, clear-path bound, and distance-dominance rejection rules",
            .symbols = "cell,wall,move,bounds,component,distance,local_topology,clear_path_upper_bound",
            .axioms = "illegal, component-trap, bound-dominated, or distance-dominated moves cannot be optimal",
            .grammar = "edge_legality, blocked_next, local_dead_end_next, component_trap_next, clear_axis_dominated_next, clear_manhattan_dominated_next, distance_dominated_next",
            .max_rule_count = 10,
            .rule_feature_count = FORMAL_FEATURE_COUNT,
            .leakage_policy = NN_LEAKAGE_PRIORS_ALLOWED,
            .rule_feature_schema = FORMAL_RULE_FEATURE_SCHEMA,
            .rule_feature_schema_count = sizeof(FORMAL_RULE_FEATURE_SCHEMA) /
                                         sizeof(FORMAL_RULE_FEATURE_SCHEMA[0]),
            .max_action_count = 4,
            .enumerate_rules = enumerate_grid_rules,
            .encode_rule = encode_grid_rule,
            .check_certificate = check_grid_certificate,
            .rule_rejects = rule_rejects_not_supported,
        },
        {
            .name = "delayed_claim",
            .description = "finite-generator claim threshold plus direct and indirect trust/request margin rules",
            .symbols = "claim_feature,threshold,action,other_action,utility_margin,risk_proxy_margin",
            .axioms = "finite support checker owns safety",
            .grammar = "reject_action_if_feature_threshold_or_trust_request_direct_or_proxy_margin",
            .max_rule_count = 8,
            .rule_feature_count = FORMAL_FEATURE_COUNT,
            .leakage_policy = NN_LEAKAGE_PRIORS_ALLOWED,
            .rule_feature_schema = FORMAL_RULE_FEATURE_SCHEMA,
            .rule_feature_schema_count = sizeof(FORMAL_RULE_FEATURE_SCHEMA) /
                                         sizeof(FORMAL_RULE_FEATURE_SCHEMA[0]),
            .max_action_count = CLAIM_ACTION_COUNT,
            .enumerate_rules = enumerate_claim_rules,
            .encode_rule = encode_claim_rule,
            .check_certificate = check_claim_certificate,
            .rule_rejects = rule_rejects_not_supported,
        },
        {
            .name = "arc_agi",
            .description = "ARC 0520fde7 transformation rules",
            .symbols = "grid,color,object,mask,transform",
            .axioms = "certified transform matches train pairs and has a unique grammar-consistent test output",
            .grammar =
                "copy_left, copy_right, mask_intersection, mask_union, object_select_overlap, "
                "color_map, layer_overlap, layer_union, layer_xor, layer_left_only, layer_right_only",
            .max_rule_count = ARC_RULE_COUNT,
            .rule_feature_count = FORMAL_FEATURE_COUNT,
            .leakage_policy = NN_LEAKAGE_PRIORS_ALLOWED,
            .rule_feature_schema = FORMAL_RULE_FEATURE_SCHEMA,
            .rule_feature_schema_count = sizeof(FORMAL_RULE_FEATURE_SCHEMA) /
                                         sizeof(FORMAL_RULE_FEATURE_SCHEMA[0]),
            .max_action_count = 10,
            .enumerate_rules = enumerate_arc_rules,
            .encode_rule = encode_arc_rule,
            .check_certificate = check_arc_certificate,
            .rule_rejects = rule_rejects_not_supported,
        },
    };

    size_t capacity = 0;
    for (size_t i = 0; i < sizeof(domains) / sizeof(domains[0]); ++i) {
        capacity += domains[i].max_rule_count;
    }
    NNRule *rules = calloc(capacity, sizeof(*rules));
    nn_real *features = calloc(capacity * FORMAL_FEATURE_COUNT, sizeof(*features));
    nn_real *targets = calloc(capacity, sizeof(*targets));
    nn_real *outputs = calloc(capacity, sizeof(*outputs));
    if (rules == NULL || features == NULL || targets == NULL || outputs == NULL) {
        free(rules);
        free(features);
        free(targets);
        free(outputs);
        return EXIT_FAILURE;
    }

    size_t count = 0;
    for (size_t i = 0; i < sizeof(domains) / sizeof(domains[0]); ++i) {
        if (!collect_domain_rules(&domains[i], rules, capacity, features, targets, &count)) {
            free(rules);
            free(features);
            free(targets);
            free(outputs);
            return EXIT_FAILURE;
        }
    }

    const size_t layers[] = {FORMAL_FEATURE_COUNT, 12, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 20260516u;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        free(rules);
        free(features);
        free(targets);
        free(outputs);
        return EXIT_FAILURE;
    }

    nn_real loss = nn_train_batched(net,
                                    features,
                                    targets,
                                    count,
                                    count,
                                    (nn_real)0.05,
                                    200,
                                    0);
    nn_forward_batch(net, features, outputs, count);
    size_t correct_side = 0;
    for (size_t i = 0; i < count; ++i) {
        int predicted = outputs[i] >= (nn_real)0.5;
        int target = targets[i] >= (nn_real)0.5;
        correct_side += predicted == target ? 1u : 0u;
    }
    printf("{\"schema\":\"crlp.symbolic_rule_proposer.summary.v1\","
           "\"domains\":%zu,\"rules\":%zu,\"feature_count\":%u,"
           "\"loss\":%.8f,\"threshold_accuracy\":%.8f}\n",
           sizeof(domains) / sizeof(domains[0]),
           count,
           FORMAL_FEATURE_COUNT,
           (double)loss,
           count == 0 ? 0.0 : (double)correct_side / (double)count);

    nn_free(net);
    free(rules);
    free(features);
    free(targets);
    free(outputs);
    return EXIT_SUCCESS;
}
