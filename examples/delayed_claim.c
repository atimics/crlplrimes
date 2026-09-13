#include "nn.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DC_STATE_FEATURES = 6,
    DC_ACTION_COUNT = 4,
    DC_FEATURE_COUNT = DC_STATE_FEATURES + DC_ACTION_COUNT,
    DC_TRAIN_SAMPLES = 4000,
    DC_VALIDATION_SAMPLES = 1000,
    DC_TEST_SAMPLES = 1000,
    DC_TRAIN_SEED_START = 1000,
    DC_VALIDATION_SEED_START = 5000,
    DC_TEST_SEED_START = 900000,
    DC_SHIFTED_TEST_SEED_START = 1900000,
    DC_UNCERTAINTY_TEST_SEED_START = 2900000,
    DC_UNCERTAINTY_SCAN_LIMIT = 200000,
    DC_REPLAY_EPOCHS = 6,
    DC_CALIBRATION_BINS = 10,
    DC_RULE_FEATURE_COUNT = 13,
};

/* Configurable evaluation band for independent generated-test replication. */
static uint64_t dc_eval_test_seed_start = DC_TEST_SEED_START;

static const NNFeatureSchema DELAYED_CLAIM_SCORER_FEATURES[] = {
    {"claim_reliability", NN_FEATURE_STATE_OBSERVATION, 1},
    {"claim_evidence", NN_FEATURE_STATE_OBSERVATION, 1},
    {"claim_contradiction", NN_FEATURE_STATE_OBSERVATION, 1},
    {"claim_age", NN_FEATURE_STATE_OBSERVATION, 1},
    {"claim_history", NN_FEATURE_STATE_OBSERVATION, 1},
    {"claim_noise", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_trust", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_ignore", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_request_more_evidence", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_act_cautiously", NN_FEATURE_STATE_OBSERVATION, 1},
};
_Static_assert(sizeof(DELAYED_CLAIM_SCORER_FEATURES) / sizeof(DELAYED_CLAIM_SCORER_FEATURES[0]) ==
                   DC_FEATURE_COUNT,
               "delayed-claim feature schema must match scorer feature count");

static const NNExperimentSpec DELAYED_CLAIM_EXPERIMENT_SPEC = {
    .task_name = "delayed_claim",
    .role = "finite delayed-feedback decision task for trace replay and verifier-RL calibration",
    .state_space = "six observable claim/evidence features generated from a deterministic simulator",
    .state_distribution = "fixed train/test seed ranges over the same hidden-rule generator",
    .action_space = "trust, ignore, request_more_evidence, act_cautiously",
    .candidate_enumerator = "enumerate all four actions for every claim state",
    .observable_features =
        "observable claim features plus action one-hot; hidden outcome and utilities are excluded",
    .leakage_policy = NN_LEAKAGE_FORBIDDEN,
    .neural_scorer = "shared MLP scores one claim/action row and softmax ranks actions",
    .grounding_authority = NN_GROUNDING_DETERMINISTIC_REPLAY,
    .grounding_operator = "deterministic hidden-rule simulator replay",
    .exact_verifier = "deterministic hidden rule computes action utilities and the best action",
    .verifier_feedback = "target action, utility, proposal steps, calibration bins, replay loss",
    .candidate_grammar = "four fixed claim actions with threshold and risk-margin rejection predicates",
    .outcome_row_schema = "crlp.trace.v1 replay rows plus delayed-claim evaluation summaries",
    .claim_scope = "finite generated delayed-feedback action ranking under declared seed bands",
    .known_non_claims =
        "does not claim behavior for real-world claim adjudication or unobserved hidden-rule families",
    .rule_grammar =
        "threshold predicates plus direct utility-margin and indirect risk-margin predicates over trust/request actions",
    .certificate_checker =
        "finite-generator safety check rejects rules that remove optimal actions on train+validation support",
    .gate_executor = "accepted predicates remove certified-unsafe actions before neural residual scoring",
    .proposer_inputs =
        "structural predicate descriptors: rule kind, feature index, threshold/margin bucket, action id, optional other-action id, polarity",
    .proposer_targets = "empirical support, marginal support, and false-rejection diagnostics from verifier labels",
    .proposer_mode = NN_PROPOSER_SELECTION,
    .baseline_orderings = "neural, closed-loop residual, random, grammar-order, structural heuristic, validation oracle",
    .generalization_axis = "train on one seed band and evaluate shifted seed bands plus harder uncertainty bands",
    .trace_schema = NN_TRACE_SCHEMA,
    .replay_protocol = "stable action-id JSONL traces train replay successors with soft and hard target weights",
    .rl_protocol =
        "supervised scorer, strict verifier policy gradient, and KL/entropy-regularized verifier policy gradient",
    .features = DELAYED_CLAIM_SCORER_FEATURES,
    .feature_count =
        sizeof(DELAYED_CLAIM_SCORER_FEATURES) / sizeof(DELAYED_CLAIM_SCORER_FEATURES[0]),
};

static const nn_real DC_REPLAY_LEARNING_RATE = (nn_real)0.008;
static const nn_real DC_REPLAY_HARD_WEIGHT = (nn_real)1.0;
static const nn_real DC_REPLAY_SOFT_WEIGHT = (nn_real)0.35;

typedef struct {
    nn_real state[DC_STATE_FEATURES];
    size_t target;
    nn_real utility[DC_ACTION_COUNT];
} ClaimState;

typedef struct {
    size_t samples;
    size_t correct;
    size_t false_rejections;
    size_t support_fallbacks;
    double nll;
    double ece;
    double utility;
    double proposal_steps;
    double baseline_steps;
    double candidate_count;
    double rejected_actions;
    size_t bin_count[DC_CALIBRATION_BINS];
    double bin_confidence[DC_CALIBRATION_BINS];
    double bin_accuracy[DC_CALIBRATION_BINS];
} ClaimMetrics;

typedef enum {
    CLAIM_PROPOSER_NEURAL,
    CLAIM_PROPOSER_CLOSED_LOOP,
    CLAIM_PROPOSER_RANDOM,
    CLAIM_PROPOSER_GRAMMAR,
    CLAIM_PROPOSER_STRUCTURAL,
    CLAIM_PROPOSER_VALIDATION_ORACLE,
} ClaimProposerMode;

typedef enum {
    CLAIM_RULE_THRESHOLD,
    CLAIM_RULE_UTILITY_MARGIN,
    CLAIM_RULE_RISK_MARGIN,
} ClaimRuleKind;

typedef enum {
    CLAIM_RULE_FAMILY_DIRECT,
    CLAIM_RULE_FAMILY_INDIRECT,
    CLAIM_RULE_FAMILY_ALL,
} ClaimRuleFamily;

typedef struct {
    size_t id;
    ClaimRuleKind kind;
    size_t action;
    size_t feature;
    size_t other_action;
    int direction;
    double threshold;
    double margin;
    int certified;
    size_t train_rejects;
    size_t train_false_rejections;
    size_t validation_rejects;
    size_t validation_false_rejections;
    size_t test_rejects;
    size_t test_false_rejections;
    double structural_score;
    double neural_score;
    double order_score;
} ClaimRule;

typedef struct {
    ClaimProposerMode mode;
    const char *label;
    const char *selection_strategy;
    ClaimRuleFamily rule_family;
    size_t max_rules;
    uint64_t seed;
} ClaimRuleSearchConfig;

typedef struct {
    const char *label;
    uint64_t seed_start;
    size_t samples;
    double observation_noise;
    double uncertainty_band;
    size_t scan_limit;
    double support_fallback_margin;
} ClaimEvalConfig;

typedef struct {
    size_t accepted_rules;
    size_t false_rejections;
    size_t train_rejects;
    size_t validation_rejects;
    size_t test_rejects;
    double direct_exact;
    double expanded_exact;
    double expanded_saved_pct;
    uint64_t rule_set_hash;
} ClaimRuleGateMetrics;

static uint32_t dc_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static nn_real dc_rng_unit(uint32_t *state)
{
    return (nn_real)((double)dc_rng_next(state) / (double)UINT32_MAX);
}

static double dc_safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

static double dc_clamp01(double value)
{
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

static double claim_truth_score_from_features(const nn_real *features)
{
    double reliability = (double)features[0];
    double evidence = (double)features[1];
    double contradiction = (double)features[2];
    double age = (double)features[3];
    double history = (double)features[4];
    double noise = (double)features[5];
    return 0.45 * reliability + 0.35 * evidence + 0.20 * history -
           0.30 * contradiction - 0.15 * noise - 0.10 * age;
}

static void compute_claim_observable_utilities(const nn_real *features,
                                               nn_real utility[DC_ACTION_COUNT])
{
    double contradiction = (double)features[2];
    double age = (double)features[3];
    double noise = (double)features[5];
    double truth_score = claim_truth_score_from_features(features);
    int resolves_true = truth_score >= 0.38;
    double uncertainty = fabs(truth_score - 0.38);

    utility[0] = resolves_true ? (nn_real)2.0 : (nn_real)-2.5;
    utility[1] = resolves_true ? (nn_real)-0.3 : (nn_real)0.3;
    utility[2] = (nn_real)(0.95 - 0.6 * uncertainty - 0.25 * age);
    utility[3] = resolves_true ? (nn_real)1.1 : (nn_real)-0.2;
    if (contradiction > 0.70 || noise > 0.80) {
        utility[2] += (nn_real)0.35;
        utility[0] -= (nn_real)0.75;
    }
}

static void compute_claim_risk_proxy_scores(const nn_real *features,
                                            double proxy[DC_ACTION_COUNT])
{
    double contradiction = (double)features[2];
    double age = (double)features[3];
    double noise = (double)features[5];
    double truth_score = claim_truth_score_from_features(features);
    double high_risk_excess = fmax(0.0, contradiction - 0.70) +
                              fmax(0.0, noise - 0.80);

    memset(proxy, 0, DC_ACTION_COUNT * sizeof(*proxy));
    proxy[0] = truth_score - 0.45 * high_risk_excess - 0.04 * age;
    proxy[2] = 0.38 - truth_score + 0.35 * high_risk_excess +
               0.08 * age + 0.06 * contradiction + 0.03 * noise;
}

static double claim_uncertainty_from_state(const ClaimState *state)
{
    return fabs(claim_truth_score_from_features(state->state) - 0.38);
}

static void perturb_claim_observation(ClaimState *state,
                                      uint64_t seed,
                                      double observation_noise)
{
    if (state == NULL || observation_noise <= 0.0) {
        return;
    }
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ 0xa5a5a5a5u);
    for (size_t i = 0; i < DC_STATE_FEATURES; ++i) {
        double delta = observation_noise * (2.0 * (double)dc_rng_unit(&rng) - 1.0);
        state->state[i] = (nn_real)dc_clamp01((double)state->state[i] + delta);
    }
}

static ClaimState generate_claim_state(uint64_t seed)
{
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ 0x9e3779b9u);
    ClaimState state;
    for (size_t i = 0; i < DC_STATE_FEATURES; ++i) {
        state.state[i] = dc_rng_unit(&rng);
    }

    compute_claim_observable_utilities(state.state, state.utility);
    state.target = 0;
    for (size_t action = 1; action < DC_ACTION_COUNT; ++action) {
        if (state.utility[action] > state.utility[state.target]) {
            state.target = action;
        }
    }
    return state;
}

static void fill_claim_candidate_features(nn_real *candidate_inputs, const ClaimState *state)
{
    for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
        nn_real *row = &candidate_inputs[action * DC_FEATURE_COUNT];
        memcpy(row, state->state, DC_STATE_FEATURES * sizeof(*row));
        for (size_t i = 0; i < DC_ACTION_COUNT; ++i) {
            row[DC_STATE_FEATURES + i] = action == i ? (nn_real)1 : (nn_real)0;
        }
    }
}

static uint32_t claim_action_id_for_action(size_t action)
{
    return (uint32_t)action;
}

static int claim_action_from_action_id(uint32_t action_id, size_t *out_action)
{
    if (action_id >= DC_ACTION_COUNT || out_action == NULL) {
        return 0;
    }

    *out_action = (size_t)action_id;
    return 1;
}

static void fill_claim_action_ids(uint32_t *action_ids)
{
    for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
        action_ids[action] = claim_action_id_for_action(action);
    }
}

static size_t claim_row_for_action_id(const uint32_t *action_ids,
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

static int claim_rule_rejects_state(const ClaimRule *rule,
                                    const ClaimState *state,
                                    size_t action)
{
    if (rule == NULL || state == NULL || action != rule->action) {
        return 0;
    }

    if (rule->kind == CLAIM_RULE_THRESHOLD) {
        if (rule->feature >= DC_STATE_FEATURES) {
            return 0;
        }
        double value = (double)state->state[rule->feature];
        return rule->direction > 0 ? value >= rule->threshold : value <= rule->threshold;
    }

    if (rule->kind == CLAIM_RULE_UTILITY_MARGIN) {
        if (rule->other_action >= DC_ACTION_COUNT || rule->other_action == rule->action) {
            return 0;
        }
        nn_real utility[DC_ACTION_COUNT];
        compute_claim_observable_utilities(state->state, utility);
        return (double)utility[rule->other_action] > (double)utility[rule->action] + rule->margin;
    }

    if (rule->kind == CLAIM_RULE_RISK_MARGIN) {
        if (rule->other_action >= DC_ACTION_COUNT || rule->other_action == rule->action) {
            return 0;
        }
        double proxy[DC_ACTION_COUNT];
        compute_claim_risk_proxy_scores(state->state, proxy);
        return proxy[rule->other_action] > proxy[rule->action] + rule->margin;
    }

    return 0;
}

static size_t enumerate_claim_rule_candidates(ClaimRule **out_rules, ClaimRuleFamily family)
{
    static const double thresholds[] = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    };
    static const double direct_margins[] = {0.0, 0.025, 0.05, 0.10, 0.20};
    static const double risk_margins[] = {0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0};
    size_t threshold_count = DC_ACTION_COUNT * DC_STATE_FEATURES * 2u *
                             (sizeof(thresholds) / sizeof(thresholds[0]));
    size_t utility_margin_count =
        (family == CLAIM_RULE_FAMILY_INDIRECT)
            ? 0u
            : 2u * (sizeof(direct_margins) / sizeof(direct_margins[0]));
    size_t risk_margin_count =
        (family == CLAIM_RULE_FAMILY_DIRECT)
            ? 0u
            : 2u * (sizeof(risk_margins) / sizeof(risk_margins[0]));
    size_t count = threshold_count + utility_margin_count + risk_margin_count;
    ClaimRule *rules = calloc(count, sizeof(*rules));
    if (rules == NULL) {
        *out_rules = NULL;
        return 0;
    }

    size_t id = 0;
    for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
        for (size_t feature = 0; feature < DC_STATE_FEATURES; ++feature) {
            for (size_t direction_index = 0; direction_index < 2; ++direction_index) {
                int direction = direction_index == 0 ? -1 : 1;
                for (size_t threshold_index = 0;
                     threshold_index < sizeof(thresholds) / sizeof(thresholds[0]);
                     ++threshold_index) {
                    ClaimRule *rule = &rules[id];
                    rule->id = id;
                    rule->kind = CLAIM_RULE_THRESHOLD;
                    rule->action = action;
                    rule->feature = feature;
                    rule->other_action = DC_ACTION_COUNT;
                    rule->direction = direction;
                    rule->threshold = thresholds[threshold_index];
                    ++id;
                }
            }
        }
    }

    if (family != CLAIM_RULE_FAMILY_INDIRECT) {
        for (size_t pair_index = 0; pair_index < 2; ++pair_index) {
            size_t action = pair_index == 0 ? 0u : 2u;
            size_t other_action = pair_index == 0 ? 2u : 0u;
            for (size_t margin_index = 0;
                 margin_index < sizeof(direct_margins) / sizeof(direct_margins[0]);
                 ++margin_index) {
                ClaimRule *rule = &rules[id];
                rule->id = id;
                rule->kind = CLAIM_RULE_UTILITY_MARGIN;
                rule->action = action;
                rule->feature = DC_STATE_FEATURES;
                rule->other_action = other_action;
                rule->direction = 1;
                rule->threshold = 0.0;
                rule->margin = direct_margins[margin_index];
                ++id;
            }
        }
    }

    if (family != CLAIM_RULE_FAMILY_DIRECT) {
        for (size_t pair_index = 0; pair_index < 2; ++pair_index) {
            size_t action = pair_index == 0 ? 0u : 2u;
            size_t other_action = pair_index == 0 ? 2u : 0u;
            for (size_t margin_index = 0;
                 margin_index < sizeof(risk_margins) / sizeof(risk_margins[0]);
                 ++margin_index) {
                ClaimRule *rule = &rules[id];
                rule->id = id;
                rule->kind = CLAIM_RULE_RISK_MARGIN;
                rule->action = action;
                rule->feature = DC_STATE_FEATURES;
                rule->other_action = other_action;
                rule->direction = 1;
                rule->threshold = 0.0;
                rule->margin = risk_margins[margin_index];
                ++id;
            }
        }
    }

    *out_rules = rules;
    return count;
}

static void audit_claim_rule_on_split(ClaimRule *rule,
                                      uint64_t seed_start,
                                      size_t samples,
                                      size_t *out_rejects,
                                      size_t *out_false_rejections)
{
    size_t rejects = 0;
    size_t false_rejections = 0;
    for (size_t i = 0; i < samples; ++i) {
        ClaimState state = generate_claim_state(seed_start + (uint64_t)i);
        if (!claim_rule_rejects_state(rule, &state, rule->action)) {
            continue;
        }
        if (rule->action == state.target) {
            ++false_rejections;
        } else {
            ++rejects;
        }
    }
    *out_rejects = rejects;
    *out_false_rejections = false_rejections;
}

static void audit_claim_rules(ClaimRule *rules, size_t rule_count)
{
    for (size_t i = 0; i < rule_count; ++i) {
        ClaimRule *rule = &rules[i];
        audit_claim_rule_on_split(rule,
                                  DC_TRAIN_SEED_START,
                                  DC_TRAIN_SAMPLES,
                                  &rule->train_rejects,
                                  &rule->train_false_rejections);
        audit_claim_rule_on_split(rule,
                                  DC_VALIDATION_SEED_START,
                                  DC_VALIDATION_SAMPLES,
                                  &rule->validation_rejects,
                                  &rule->validation_false_rejections);
        audit_claim_rule_on_split(rule,
                                  DC_TEST_SEED_START,
                                  DC_TEST_SAMPLES,
                                  &rule->test_rejects,
                                  &rule->test_false_rejections);
        rule->certified = rule->train_rejects != 0 &&
                          rule->train_false_rejections == 0 &&
                          rule->validation_false_rejections == 0;
        if (rule->kind == CLAIM_RULE_UTILITY_MARGIN) {
            double margin_prior = 1.0 / (1.0 + 10.0 * rule->margin);
            rule->structural_score = 2.0 + 0.25 * margin_prior;
        } else if (rule->kind == CLAIM_RULE_RISK_MARGIN) {
            double margin_prior = 1.0 / (1.0 + 2.0 * rule->margin);
            rule->structural_score = 1.55 + 0.25 * margin_prior;
        } else {
            double action_prior = (rule->action == 1 || rule->action == 3) ? 1.0 : 0.0;
            double broadness = rule->direction > 0 ? 1.0 - rule->threshold : rule->threshold;
            rule->structural_score = action_prior + 0.25 * broadness;
        }
    }
}

static void encode_claim_rule_features(const ClaimRule *rule, double features[DC_RULE_FEATURE_COUNT])
{
    memset(features, 0, DC_RULE_FEATURE_COUNT * sizeof(*features));
    features[0] = 1.0;
    features[1] = (double)rule->action / (double)(DC_ACTION_COUNT - 1u);
    if (rule->kind == CLAIM_RULE_THRESHOLD) {
        features[2] = (double)rule->feature / (double)(DC_STATE_FEATURES - 1u);
        features[3] = rule->direction > 0 ? 1.0 : 0.0;
        features[4] = rule->threshold;
        features[5] = rule->direction > 0 ? 1.0 - rule->threshold : rule->threshold;
        features[6] = (rule->threshold <= 0.0 || rule->threshold >= 1.0) ? 1.0 : 0.0;
    }
    features[7] = (rule->action == 1 || rule->action == 3) ? 1.0 : 0.0;
    features[8] = rule->kind == CLAIM_RULE_THRESHOLD ? 1.0 : 0.0;
    features[9] = rule->kind == CLAIM_RULE_UTILITY_MARGIN ? 1.0 : 0.0;
    features[10] = rule->kind == CLAIM_RULE_RISK_MARGIN ? 1.0 : 0.0;
    features[11] = rule->other_action < DC_ACTION_COUNT
                       ? (double)rule->other_action / (double)(DC_ACTION_COUNT - 1u)
                       : 0.0;
    features[12] = rule->margin;
}

static void score_claim_rules_with_neural_prior(ClaimRule *rules, size_t rule_count)
{
    double weights[DC_RULE_FEATURE_COUNT] = {0};
    const double learning_rate = 0.08;
    for (size_t epoch = 0; epoch < 400; ++epoch) {
        for (size_t i = 0; i < rule_count; ++i) {
            double features[DC_RULE_FEATURE_COUNT];
            encode_claim_rule_features(&rules[i], features);
            double target = rules[i].certified
                                ? (double)rules[i].train_rejects / (double)DC_TRAIN_SAMPLES
                                : 0.0;
            double prediction = 0.0;
            for (size_t j = 0; j < DC_RULE_FEATURE_COUNT; ++j) {
                prediction += weights[j] * features[j];
            }
            double error = prediction - target;
            for (size_t j = 0; j < DC_RULE_FEATURE_COUNT; ++j) {
                weights[j] -= learning_rate * error * features[j];
            }
        }
    }
    for (size_t i = 0; i < rule_count; ++i) {
        double features[DC_RULE_FEATURE_COUNT];
        encode_claim_rule_features(&rules[i], features);
        rules[i].neural_score = 0.0;
        for (size_t j = 0; j < DC_RULE_FEATURE_COUNT; ++j) {
            rules[i].neural_score += weights[j] * features[j];
        }
    }
}

static int compare_claim_rule_order(const void *left, const void *right)
{
    const ClaimRule *a = *(const ClaimRule *const *)left;
    const ClaimRule *b = *(const ClaimRule *const *)right;
    if (a->order_score < b->order_score) {
        return 1;
    }
    if (a->order_score > b->order_score) {
        return -1;
    }
    return a->id < b->id ? -1 : (a->id > b->id ? 1 : 0);
}

static uint64_t claim_rule_mix_hash(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}

static const char *claim_rule_kind_name(ClaimRuleKind kind)
{
    switch (kind) {
    case CLAIM_RULE_THRESHOLD:
        return "threshold";
    case CLAIM_RULE_UTILITY_MARGIN:
        return "utility_margin";
    case CLAIM_RULE_RISK_MARGIN:
        return "risk_margin";
    default:
        return "unknown";
    }
}

static const char *claim_rule_family_name(ClaimRuleFamily family)
{
    switch (family) {
    case CLAIM_RULE_FAMILY_DIRECT:
        return "direct";
    case CLAIM_RULE_FAMILY_INDIRECT:
        return "indirect";
    case CLAIM_RULE_FAMILY_ALL:
        return "all";
    default:
        return "unknown";
    }
}

static const char *claim_rule_grammar_version(ClaimRuleFamily family)
{
    switch (family) {
    case CLAIM_RULE_FAMILY_DIRECT:
        return "claim_threshold_margin_v1";
    case CLAIM_RULE_FAMILY_INDIRECT:
        return "claim_threshold_risk_margin_v1";
    case CLAIM_RULE_FAMILY_ALL:
        return "claim_threshold_direct_indirect_margin_v1";
    default:
        return "claim_threshold_margin_unknown";
    }
}

static int parse_claim_rule_family(const char *text, ClaimRuleFamily *out_family)
{
    if (strcmp(text, "direct") == 0 || strcmp(text, "utility") == 0) {
        *out_family = CLAIM_RULE_FAMILY_DIRECT;
    } else if (strcmp(text, "indirect") == 0 || strcmp(text, "risk") == 0) {
        *out_family = CLAIM_RULE_FAMILY_INDIRECT;
    } else if (strcmp(text, "all") == 0) {
        *out_family = CLAIM_RULE_FAMILY_ALL;
    } else {
        return 0;
    }
    return 1;
}

static const char *claim_action_name(size_t action)
{
    switch (action) {
    case 0:
        return "trust";
    case 1:
        return "ignore";
    case 2:
        return "request_more_evidence";
    case 3:
        return "act_cautiously";
    default:
        return "unknown";
    }
}

static NeuralNet *create_claim_model(unsigned int seed)
{
    static const size_t layers[] = {DC_FEATURE_COUNT, 24, 12, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static ClaimMetrics evaluate_claim_model(NeuralNet *net,
                                         uint64_t seed_start,
                                         size_t samples,
                                         nn_real *candidate_inputs,
                                         nn_real *scores,
                                         nn_real *probabilities)
{
    ClaimMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    size_t ranked[DC_ACTION_COUNT];

    for (size_t i = 0; i < samples; ++i) {
        ClaimState state = generate_claim_state(seed_start + (uint64_t)i);
        fill_claim_candidate_features(candidate_inputs, &state);
        if (!nn_score_action_rows(net,
                                  candidate_inputs,
                                  DC_ACTION_COUNT,
                                  scores,
                                  probabilities) ||
            !nn_rank_action_scores(scores, DC_ACTION_COUNT, ranked)) {
            continue;
        }

        size_t selected = ranked[0];
        size_t target_rank = DC_ACTION_COUNT;
        for (size_t rank = 0; rank < DC_ACTION_COUNT; ++rank) {
            if (ranked[rank] == state.target) {
                target_rank = rank;
                break;
            }
        }
        double confidence = (double)probabilities[selected];
        size_t bin = confidence >= 1.0 ? DC_CALIBRATION_BINS - 1
                                       : (size_t)(confidence * DC_CALIBRATION_BINS);
        if (bin >= DC_CALIBRATION_BINS) {
            bin = DC_CALIBRATION_BINS - 1;
        }

        ++metrics.samples;
        metrics.correct += selected == state.target ? 1u : 0u;
        metrics.nll += -log(dc_safe_probability((double)probabilities[state.target]));
        metrics.utility += (double)state.utility[selected];
        metrics.proposal_steps += (double)target_rank + 1.0;
        metrics.baseline_steps += (double)state.target + 1.0;
        metrics.candidate_count += (double)DC_ACTION_COUNT;
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += selected == state.target ? 1.0 : 0.0;
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= sample_count;
    metrics.utility /= sample_count;
    metrics.proposal_steps /= sample_count;
    metrics.baseline_steps /= sample_count;
    metrics.candidate_count /= sample_count;
    metrics.rejected_actions /= sample_count;
    for (size_t bin = 0; bin < DC_CALIBRATION_BINS; ++bin) {
        if (metrics.bin_count[bin] == 0) {
            continue;
        }
        double count = (double)metrics.bin_count[bin];
        double confidence = metrics.bin_confidence[bin] / count;
        double accuracy = metrics.bin_accuracy[bin] / count;
        metrics.ece += (count / sample_count) * fabs(confidence - accuracy);
        metrics.bin_confidence[bin] = confidence;
        metrics.bin_accuracy[bin] = accuracy;
    }
    return metrics;
}

static const ClaimEvalConfig DC_EVAL_CANONICAL = {
    "canonical", DC_TEST_SEED_START, DC_TEST_SAMPLES, 0.0, 0.0, DC_TEST_SAMPLES, 0.0,
};

static const ClaimEvalConfig DC_EVAL_STRESS_BANDS[] = {
    {"shifted_seed", DC_SHIFTED_TEST_SEED_START, DC_TEST_SAMPLES, 0.0, 0.0, DC_TEST_SAMPLES, 0.0},
    {"observation_noise_0p03", DC_TEST_SEED_START, DC_TEST_SAMPLES, 0.03, 0.0, DC_TEST_SAMPLES, 0.0},
    {"uncertainty_band_noise_0p03",
     DC_UNCERTAINTY_TEST_SEED_START,
     DC_TEST_SAMPLES,
     0.03,
     0.04,
     DC_UNCERTAINTY_SCAN_LIMIT,
     0.0},
};

static ClaimMetrics evaluate_claim_model_with_gate_on_band(NeuralNet *net,
                                                           const ClaimEvalConfig *eval_config,
                                                           const ClaimRule *rules,
                                                           size_t rule_count,
                                                           nn_real *candidate_inputs,
                                                           nn_real *scores,
                                                           nn_real *probabilities)
{
    ClaimMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    const ClaimEvalConfig *config =
        eval_config == NULL ? &DC_EVAL_CANONICAL : eval_config;
    size_t scanned = 0;
    uint64_t seed = config->seed_start;
    while (metrics.samples < config->samples && scanned < config->scan_limit) {
        ClaimState truth_state = generate_claim_state(seed);
        ++seed;
        ++scanned;
        if (config->uncertainty_band > 0.0 &&
            claim_uncertainty_from_state(&truth_state) > config->uncertainty_band) {
            continue;
        }

        ClaimState observed_state = truth_state;
        perturb_claim_observation(&observed_state,
                                  seed + UINT64_C(0x517cc1b727220a95),
                                  config->observation_noise);
        observed_state.target = truth_state.target;
        memcpy(observed_state.utility, truth_state.utility, sizeof(observed_state.utility));

        fill_claim_candidate_features(candidate_inputs, &observed_state);
        if (!nn_score_action_rows(net,
                                  candidate_inputs,
                                  DC_ACTION_COUNT,
                                  scores,
                                  NULL)) {
            continue;
        }

        uint8_t rejected[DC_ACTION_COUNT] = {0};
        size_t rejected_count = 0;
        int support_fallback = config->support_fallback_margin > 0.0 &&
                               claim_uncertainty_from_state(&observed_state) <=
                                   config->support_fallback_margin;
        if (support_fallback) {
            ++metrics.support_fallbacks;
        }
        for (size_t action = 0; !support_fallback && action < DC_ACTION_COUNT; ++action) {
            for (size_t rule_index = 0; rule_index < rule_count; ++rule_index) {
                if (claim_rule_rejects_state(&rules[rule_index], &observed_state, action)) {
                    rejected[action] = 1;
                    break;
                }
            }
            if (rejected[action]) {
                ++rejected_count;
            }
        }
        if (rejected[truth_state.target]) {
            ++metrics.false_rejections;
        }
        if (rejected_count >= DC_ACTION_COUNT || rejected[truth_state.target]) {
            memset(rejected, 0, sizeof(rejected));
            rejected_count = 0;
        }

        double max_score = -INFINITY;
        for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
            if (!rejected[action] && (double)scores[action] > max_score) {
                max_score = (double)scores[action];
            }
        }
        double sum = 0.0;
        for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
            if (rejected[action]) {
                probabilities[action] = (nn_real)0;
            } else {
                double value = exp((double)scores[action] - max_score);
                probabilities[action] = (nn_real)value;
                sum += value;
            }
        }
        if (sum <= 0.0 || !isfinite(sum)) {
            continue;
        }
        for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
            probabilities[action] = (nn_real)((double)probabilities[action] / sum);
        }

        size_t selected = DC_ACTION_COUNT;
        double selected_score = -INFINITY;
        for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
            if (!rejected[action] && (double)scores[action] > selected_score) {
                selected_score = (double)scores[action];
                selected = action;
            }
        }
        size_t target_rank = 1;
        for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
            if (!rejected[action] && action != truth_state.target &&
                scores[action] > scores[truth_state.target]) {
                ++target_rank;
            }
        }
        double confidence = (double)probabilities[selected];
        size_t bin = confidence >= 1.0 ? DC_CALIBRATION_BINS - 1
                                       : (size_t)(confidence * DC_CALIBRATION_BINS);
        if (bin >= DC_CALIBRATION_BINS) {
            bin = DC_CALIBRATION_BINS - 1;
        }

        ++metrics.samples;
        metrics.correct += selected == truth_state.target ? 1u : 0u;
        metrics.nll += -log(dc_safe_probability((double)probabilities[truth_state.target]));
        metrics.utility += (double)truth_state.utility[selected];
        metrics.proposal_steps += (double)target_rank;
        metrics.baseline_steps += (double)DC_ACTION_COUNT;
        metrics.candidate_count += (double)(DC_ACTION_COUNT - rejected_count);
        metrics.rejected_actions += (double)rejected_count;
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += selected == truth_state.target ? 1.0 : 0.0;
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= sample_count;
    metrics.utility /= sample_count;
    metrics.proposal_steps /= sample_count;
    metrics.baseline_steps /= sample_count;
    metrics.candidate_count /= sample_count;
    metrics.rejected_actions /= sample_count;
    for (size_t bin = 0; bin < DC_CALIBRATION_BINS; ++bin) {
        if (metrics.bin_count[bin] == 0) {
            continue;
        }
        double count = (double)metrics.bin_count[bin];
        double confidence = metrics.bin_confidence[bin] / count;
        double accuracy = metrics.bin_accuracy[bin] / count;
        metrics.ece += (count / sample_count) * fabs(confidence - accuracy);
        metrics.bin_confidence[bin] = confidence;
        metrics.bin_accuracy[bin] = accuracy;
    }
    return metrics;
}

static int train_claim_model(NeuralNet *net,
                             uint64_t seed_start,
                             size_t samples,
                             size_t epochs,
                             nn_real *candidate_inputs)
{
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < samples; ++i) {
            ClaimState state = generate_claim_state(seed_start + (uint64_t)i);
            fill_claim_candidate_features(candidate_inputs, &state);
            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       DC_ACTION_COUNT,
                                                       state.target,
                                                       (nn_real)0.01);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }
        printf("delayed_claim epoch %zu loss=%.4f samples=%zu\n",
               epoch + 1,
               loss / (double)samples,
               samples);
    }
    return 1;
}

static void print_claim_metrics(const char *label, const char *hash, ClaimMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    double saved = metrics.baseline_steps - metrics.proposal_steps;
    double saved_pct = metrics.baseline_steps <= 0.0 ? 0.0 : 100.0 * saved / metrics.baseline_steps;
    printf("[%s] hash=%s accuracy=%.2f%% nll=%.4f ece=%.4f decision_utility=%.4f "
           "proposal_steps=%.2f baseline_steps=%.2f saved=%.2f (%.1f%%) samples=%zu\n",
           label,
           hash == NULL ? "unknown" : hash,
           100.0 * (double)metrics.correct / samples,
           metrics.nll,
           metrics.ece,
           metrics.utility,
           metrics.proposal_steps,
           metrics.baseline_steps,
           saved,
           saved_pct,
           metrics.samples);
}

static int save_claim_checkpoint(NeuralNet *net,
                                 const char *path,
                                 const char *parent_hash,
                                 const char *notes,
                                 char out_hash[65])
{
    NNCheckpointMeta meta;
    nn_checkpoint_meta_init(&meta, 1);
    if (!nn_checkpoint_meta_set_parent_model_hash(&meta,
                                                  parent_hash == NULL ? "unknown" : parent_hash) ||
        !nn_checkpoint_meta_set_training_trace_root(&meta, "delayed_claim") ||
        !nn_checkpoint_meta_set_notes(&meta, notes) ||
        !nn_checkpoint_meta_set_feature_set(&meta, "claim-state-action-v1")) {
        return 0;
    }
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

static uint64_t claim_state_hash(uint64_t seed)
{
    uint64_t hash = 1469598103934665603ull;
    hash ^= seed;
    hash *= 1099511628211ull;
    return hash;
}

static int emit_claim_trace(NeuralNet *net,
                            const char *path,
                            const char *model_hash,
                            uint64_t seed_start,
                            size_t samples,
                            nn_real *candidate_inputs,
                            nn_real *scores,
                            nn_real *probabilities)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    uint32_t action_ids[DC_ACTION_COUNT];
    fill_claim_action_ids(action_ids);
    size_t ranked[DC_ACTION_COUNT];
    int ok = 1;
    for (size_t i = 0; ok && i < samples; ++i) {
        uint64_t seed = seed_start + (uint64_t)i;
        ClaimState state = generate_claim_state(seed);
        uint32_t target_action_id = claim_action_id_for_action(state.target);
        size_t target_index = claim_row_for_action_id(action_ids,
                                                      DC_ACTION_COUNT,
                                                      target_action_id);
        fill_claim_candidate_features(candidate_inputs, &state);
        ok = nn_score_action_rows(net,
                                  candidate_inputs,
                                  DC_ACTION_COUNT,
                                  scores,
                                  probabilities) &&
             nn_rank_action_scores(scores, DC_ACTION_COUNT, ranked);
        if (!ok) {
            break;
        }
        if (target_index == DC_ACTION_COUNT) {
            ok = 0;
            break;
        }
        size_t proposal_steps = DC_ACTION_COUNT;
        for (size_t rank = 0; rank < DC_ACTION_COUNT; ++rank) {
            if (action_ids[ranked[rank]] == target_action_id) {
                proposal_steps = rank + 1;
                break;
            }
        }
        size_t selected_action = DC_ACTION_COUNT;
        if (!claim_action_from_action_id(action_ids[ranked[0]], &selected_action)) {
            ok = 0;
            break;
        }

        char cost_json[128];
        char extra_json[128];
        snprintf(cost_json,
                 sizeof(cost_json),
                 "{\"decision_utility\":%.6f,\"proposal_steps\":%zu,"
                 "\"baseline_steps\":%zu}",
                 (double)state.utility[selected_action],
                 proposal_steps,
                 state.target + 1);
        snprintf(extra_json,
                 sizeof(extra_json),
                 "\"decision_utility\":%.6f,\"proposal_steps\":%zu,"
                 "\"baseline_steps\":%zu",
                 (double)state.utility[selected_action],
                 proposal_steps,
                 state.target + 1);

        NNTraceRecord record;
        memset(&record, 0, sizeof(record));
        record.task_name = "delayed_claim";
        record.model_hash_hex = model_hash == NULL ? "unknown" : model_hash;
        record.state_hash = claim_state_hash(seed);
        record.feature_encoder_version = 1;
        record.candidate_count = DC_ACTION_COUNT;
        record.action_ids = action_ids;
        record.scores = scores;
        record.probabilities = probabilities;
        record.selected_index = ranked[0];
        record.target_index = target_index;
        record.loss =
            (nn_real)(-log(dc_safe_probability((double)probabilities[target_index])));
        record.verifier_result =
            action_ids[ranked[0]] == target_action_id ? "correct" : "corrected";
        record.verifier_cost_json = cost_json;
        record.seed = seed;
        record.extra_json = extra_json;
        record.features = &candidate_inputs[target_index * DC_FEATURE_COUNT];
        record.feature_count = DC_FEATURE_COUNT;
        ok = nn_trace_write_jsonl(fp, &record);
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (ok) {
        printf("wrote delayed_claim trace: %s rows=%zu\n", path, samples);
    }
    return ok;
}

typedef struct {
    uint64_t *seeds;
    uint32_t *target_action_ids;
    nn_real *reference_probabilities;
    size_t count;
    size_t capacity;
    char parent_hash[65];
} ClaimTrace;

typedef struct {
    const char *trace_path;
    const char *out_path;
    size_t epochs;
    nn_real learning_rate;
    nn_real hard_weight;
    nn_real soft_weight;
} ClaimReplayConfig;

static int claim_trace_append(ClaimTrace *trace,
                              uint64_t seed,
                              uint32_t target_action_id,
                              const nn_real *reference_probabilities)
{
    if (trace->count == trace->capacity) {
        size_t next_capacity = trace->capacity == 0 ? 1024 : trace->capacity * 2;
        uint64_t *next_seeds = calloc(next_capacity, sizeof(*next_seeds));
        uint32_t *next_target_action_ids =
            calloc(next_capacity, sizeof(*next_target_action_ids));
        nn_real *next_reference_probabilities =
            calloc(next_capacity * DC_ACTION_COUNT, sizeof(*next_reference_probabilities));
        if (next_seeds == NULL || next_target_action_ids == NULL ||
            next_reference_probabilities == NULL) {
            free(next_seeds);
            free(next_target_action_ids);
            free(next_reference_probabilities);
            return 0;
        }
        if (trace->count != 0) {
            memcpy(next_seeds, trace->seeds, trace->count * sizeof(*next_seeds));
            memcpy(next_target_action_ids,
                   trace->target_action_ids,
                   trace->count * sizeof(*next_target_action_ids));
            memcpy(next_reference_probabilities,
                   trace->reference_probabilities,
                   trace->count * DC_ACTION_COUNT * sizeof(*next_reference_probabilities));
        }
        free(trace->seeds);
        free(trace->target_action_ids);
        free(trace->reference_probabilities);
        trace->seeds = next_seeds;
        trace->target_action_ids = next_target_action_ids;
        trace->reference_probabilities = next_reference_probabilities;
        trace->capacity = next_capacity;
    }
    trace->seeds[trace->count] = seed;
    trace->target_action_ids[trace->count] = target_action_id;
    memcpy(&trace->reference_probabilities[trace->count * DC_ACTION_COUNT],
           reference_probabilities,
           DC_ACTION_COUNT * sizeof(*reference_probabilities));
    ++trace->count;
    return 1;
}

static void claim_trace_free(ClaimTrace *trace)
{
    free(trace->seeds);
    free(trace->target_action_ids);
    free(trace->reference_probabilities);
    memset(trace, 0, sizeof(*trace));
}

static int load_claim_trace(const char *path, ClaimTrace *trace);

static int delayed_claim_reordered_trace_self_test(void)
{
    const char *path = "/tmp/crlplrimes_delayed_claim_reordered_trace.jsonl";
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp,
            "{\"schema\":\"crlp.trace.v1\",\"trace_version\":1,"
            "\"action_id_contract\":\"stable_unique_action_ids\",\"task\":\"delayed_claim\","
            "\"state_hash\":\"0000000000000001\",\"model_hash\":\"fixture\","
            "\"feature_encoder_version\":1,\"candidate_count\":4,"
            "\"action_ids\":[2,0,3,1],\"scores\":[2,0,-1,-2],"
            "\"probabilities\":[0.7,0.1,0.1,0.1],"
            "\"selected_index\":0,\"target_index\":0,\"loss\":0.1,"
            "\"verifier_result\":\"correct\",\"verifier_cost\":{},\"seed\":1000}\n");
    if (fclose(fp) != 0) {
        return 0;
    }

    ClaimTrace trace;
    int ok = load_claim_trace(path, &trace);
    if (ok) {
        uint32_t current_action_ids[DC_ACTION_COUNT];
        nn_real row_reference_probabilities[DC_ACTION_COUNT];
        fill_claim_action_ids(current_action_ids);
        for (size_t row = 0; row < DC_ACTION_COUNT; ++row) {
            size_t action = DC_ACTION_COUNT;
            ok = ok && claim_action_from_action_id(current_action_ids[row], &action);
            if (ok) {
                row_reference_probabilities[row] =
                    trace.reference_probabilities[action];
            }
        }
        ok = ok && trace.count == 1 && trace.target_action_ids[0] == 2 &&
             claim_row_for_action_id(current_action_ids,
                                     DC_ACTION_COUNT,
                                     trace.target_action_ids[0]) == 2 &&
             fabs((double)row_reference_probabilities[2] - 0.7) < 1e-6;
    }
    claim_trace_free(&trace);
    remove(path);
    printf("%s delayed_claim reordered trace action-id replay\n", ok ? "ok" : "FAIL");
    return ok;
}

static int load_claim_trace(const char *path, ClaimTrace *trace)
{
    memset(trace, 0, sizeof(*trace));
    strcpy(trace->parent_hash, "unknown");
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    char line[8192];
    uint32_t action_ids[DC_ACTION_COUNT];
    nn_real scores[DC_ACTION_COUNT];
    nn_real probabilities[DC_ACTION_COUNT];
    int ok = 1;
    while (ok && fgets(line, sizeof(line), fp) != NULL) {
        NNParsedTraceRecord record;
        if (!nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  DC_ACTION_COUNT)) {
            ok = 0;
            break;
        }
        if (strcmp(record.task_name, "delayed_claim") != 0) {
            continue;
        }
        nn_real reference_probabilities[DC_ACTION_COUNT] = {0};
        uint8_t seen_actions[DC_ACTION_COUNT] = {0};
        for (size_t row = 0; row < record.candidate_count; ++row) {
            size_t action = DC_ACTION_COUNT;
            if (!claim_action_from_action_id(action_ids[row], &action) ||
                seen_actions[action]) {
                ok = 0;
                break;
            }
            seen_actions[action] = 1;
            reference_probabilities[action] = probabilities[row];
        }
        if (!ok || record.candidate_count != DC_ACTION_COUNT ||
            record.target_index >= DC_ACTION_COUNT) {
            ok = 0;
            break;
        }
        if (strcmp(trace->parent_hash, "unknown") == 0 &&
            strcmp(record.model_hash_hex, "unknown") != 0) {
            strcpy(trace->parent_hash, record.model_hash_hex);
        }
        ok = claim_trace_append(trace,
                                record.seed,
                                action_ids[record.target_index],
                                reference_probabilities);
    }
    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || trace->count == 0) {
        claim_trace_free(trace);
        return 0;
    }
    return 1;
}

static int replay_claim_trace(const ClaimReplayConfig *config)
{
    ClaimTrace trace;
    if (!load_claim_trace(config->trace_path, &trace)) {
        fprintf(stderr, "failed to load trace: %s\n", config->trace_path);
        return 0;
    }

    NeuralNet *net = create_claim_model(9090);
    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(DC_ACTION_COUNT, sizeof(*scores));
    if (net == NULL || candidate_inputs == NULL || scores == NULL) {
        nn_free(net);
        free(candidate_inputs);
        free(scores);
        claim_trace_free(&trace);
        return 0;
    }

    uint32_t current_action_ids[DC_ACTION_COUNT];
    fill_claim_action_ids(current_action_ids);
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < trace.count; ++i) {
            ClaimState state = generate_claim_state(trace.seeds[i]);
            size_t target_index = claim_row_for_action_id(current_action_ids,
                                                          DC_ACTION_COUNT,
                                                          trace.target_action_ids[i]);
            nn_real row_reference_probabilities[DC_ACTION_COUNT];

            if (target_index == DC_ACTION_COUNT) {
                nn_free(net);
                free(candidate_inputs);
                free(scores);
                claim_trace_free(&trace);
                return 0;
            }
            for (size_t row = 0; row < DC_ACTION_COUNT; ++row) {
                size_t action = DC_ACTION_COUNT;
                if (!claim_action_from_action_id(current_action_ids[row], &action)) {
                    nn_free(net);
                    free(candidate_inputs);
                    free(scores);
                    claim_trace_free(&trace);
                    return 0;
                }
                row_reference_probabilities[row] =
                    trace.reference_probabilities[i * DC_ACTION_COUNT + action];
            }

            fill_claim_candidate_features(candidate_inputs, &state);
            if (!nn_score_action_rows(net, candidate_inputs, DC_ACTION_COUNT, scores, NULL)) {
                nn_free(net);
                free(candidate_inputs);
                free(scores);
                claim_trace_free(&trace);
                return 0;
            }
            nn_real step_loss = nn_train_softmax_scorer_policy_regularized_cached(
                net,
                DC_ACTION_COUNT,
                target_index,
                config->hard_weight,
                row_reference_probabilities,
                config->soft_weight,
                (nn_real)0,
                config->learning_rate);
            if (isnan((double)step_loss)) {
                nn_free(net);
                free(candidate_inputs);
                free(scores);
                claim_trace_free(&trace);
                return 0;
            }
            loss += (double)step_loss;
        }
        printf("delayed_claim replay epoch %zu loss=%.4f rows=%zu\n",
               epoch + 1,
               loss / (double)trace.count,
               trace.count);
    }

    char hash[65];
    int ok = save_claim_checkpoint(net,
                                   config->out_path,
                                   trace.parent_hash,
                                   "delayed_claim successor trained from soft trace replay",
                                   hash);
    if (ok) {
        printf("saved delayed_claim replay checkpoint: %s hash=%s parent=%s "
               "epochs=%zu hard_weight=%.2f soft_weight=%.2f\n",
               config->out_path,
               hash,
               trace.parent_hash,
               config->epochs,
               (double)config->hard_weight,
               (double)config->soft_weight);
    }
    nn_free(net);
    free(candidate_inputs);
    free(scores);
    claim_trace_free(&trace);
    return ok;
}

static void print_claim_reliability_bins_json(const ClaimMetrics *metrics)
{
    fputs(",\"reliability_bins\":[", stdout);
    int first = 1;
    for (size_t bin = 0; bin < DC_CALIBRATION_BINS; ++bin) {
        if (metrics->bin_count[bin] == 0) {
            continue;
        }
        printf("%s{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
               "\"confidence\":%.8f,\"accuracy\":%.8f}",
               first ? "" : ",",
               (double)bin / (double)DC_CALIBRATION_BINS,
               (double)(bin + 1) / (double)DC_CALIBRATION_BINS,
               metrics->bin_count[bin],
               metrics->bin_confidence[bin],
               metrics->bin_accuracy[bin]);
        first = 0;
    }
    fputc(']', stdout);
}

static ClaimRuleGateMetrics evaluate_claim_rule_gate(const ClaimRule *rules,
                                                     size_t rule_count,
                                                     const size_t *accepted,
                                                     size_t accepted_count)
{
    (void)rule_count;
    ClaimRuleGateMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    metrics.accepted_rules = accepted_count;
    metrics.direct_exact = (double)DC_ACTION_COUNT;
    metrics.rule_set_hash = UINT64_C(1469598103934665603);

    for (size_t i = 0; i < accepted_count; ++i) {
        const ClaimRule *rule = &rules[accepted[i]];
        metrics.train_rejects += rule->train_rejects;
        metrics.validation_rejects += rule->validation_rejects;
        metrics.test_rejects += rule->test_rejects;
        metrics.false_rejections += rule->test_false_rejections;
        metrics.rule_set_hash = claim_rule_mix_hash(metrics.rule_set_hash, (uint64_t)rule->id);
    }

    double survivors_total = 0.0;
    size_t false_rejections = 0;
    for (size_t i = 0; i < DC_TEST_SAMPLES; ++i) {
        ClaimState state = generate_claim_state(DC_TEST_SEED_START + (uint64_t)i);
        size_t rejected_count = 0;
        for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
            int rejected = 0;
            for (size_t rule_index = 0; rule_index < accepted_count; ++rule_index) {
                if (claim_rule_rejects_state(&rules[accepted[rule_index]], &state, action)) {
                    rejected = 1;
                    break;
                }
            }
            if (rejected) {
                ++rejected_count;
                if (action == state.target) {
                    ++false_rejections;
                }
            }
        }
        if (rejected_count >= DC_ACTION_COUNT) {
            rejected_count = 0;
        }
        survivors_total += (double)(DC_ACTION_COUNT - rejected_count);
    }
    metrics.false_rejections = false_rejections;
    metrics.expanded_exact = survivors_total / (double)DC_TEST_SAMPLES;
    metrics.expanded_saved_pct =
        100.0 * (metrics.direct_exact - metrics.expanded_exact) / metrics.direct_exact;
    return metrics;
}

static int select_claim_rules_ordered(ClaimRule *rules,
                                      size_t rule_count,
                                      ClaimProposerMode mode,
                                      uint64_t seed,
                                      size_t max_rules,
                                      size_t *accepted,
                                      size_t *accepted_count)
{
    ClaimRule **order = calloc(rule_count, sizeof(*order));
    uint8_t *covered = calloc((size_t)DC_VALIDATION_SAMPLES * DC_ACTION_COUNT,
                              sizeof(*covered));
    if (order == NULL || covered == NULL) {
        free(order);
        free(covered);
        return 0;
    }
    uint64_t rng = seed == 0 ? UINT64_C(20240511) : seed;
    for (size_t i = 0; i < rule_count; ++i) {
        order[i] = &rules[i];
        switch (mode) {
        case CLAIM_PROPOSER_NEURAL:
            rules[i].order_score = rules[i].neural_score;
            break;
        case CLAIM_PROPOSER_RANDOM:
            rng = rng * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
            rules[i].order_score = (double)(rng >> 11) / (double)(UINT64_MAX >> 11);
            break;
        case CLAIM_PROPOSER_GRAMMAR:
            rules[i].order_score = -(double)rules[i].id;
            break;
        case CLAIM_PROPOSER_STRUCTURAL:
            rules[i].order_score = rules[i].structural_score;
            break;
        case CLAIM_PROPOSER_VALIDATION_ORACLE:
            rules[i].order_score = (double)rules[i].validation_rejects;
            break;
        case CLAIM_PROPOSER_CLOSED_LOOP:
            rules[i].order_score = 0.0;
            break;
        }
    }
    qsort(order, rule_count, sizeof(*order), compare_claim_rule_order);

    *accepted_count = 0;
    for (size_t i = 0; i < rule_count && *accepted_count < max_rules; ++i) {
        if (!order[i]->certified) {
            continue;
        }
        size_t marginal = 0;
        for (size_t sample = 0; sample < DC_VALIDATION_SAMPLES; ++sample) {
            ClaimState state =
                generate_claim_state(DC_VALIDATION_SEED_START + (uint64_t)sample);
            size_t action = order[i]->action;
            size_t offset = sample * DC_ACTION_COUNT + action;
            if (action != state.target && !covered[offset] &&
                claim_rule_rejects_state(order[i], &state, action)) {
                ++marginal;
            }
        }
        if (marginal == 0) {
            continue;
        }
        accepted[*accepted_count] = order[i]->id;
        ++*accepted_count;
        for (size_t sample = 0; sample < DC_VALIDATION_SAMPLES; ++sample) {
            ClaimState state =
                generate_claim_state(DC_VALIDATION_SEED_START + (uint64_t)sample);
            size_t action = order[i]->action;
            if (action != state.target &&
                claim_rule_rejects_state(order[i], &state, action)) {
                covered[sample * DC_ACTION_COUNT + action] = 1;
            }
        }
    }
    free(covered);
    free(order);
    return 1;
}

static int select_claim_rules_closed_loop(ClaimRule *rules,
                                          size_t rule_count,
                                          size_t max_rules,
                                          size_t *accepted,
                                          size_t *accepted_count)
{
    uint8_t *covered = calloc((size_t)DC_VALIDATION_SAMPLES * DC_ACTION_COUNT,
                              sizeof(*covered));
    uint8_t *selected = calloc(rule_count, sizeof(*selected));
    if (covered == NULL || selected == NULL) {
        free(covered);
        free(selected);
        return 0;
    }

    *accepted_count = 0;
    while (*accepted_count < max_rules) {
        size_t best = rule_count;
        size_t best_marginal = 0;
        for (size_t rule_index = 0; rule_index < rule_count; ++rule_index) {
            if (selected[rule_index] || !rules[rule_index].certified) {
                continue;
            }
            size_t marginal = 0;
            for (size_t i = 0; i < DC_VALIDATION_SAMPLES; ++i) {
                ClaimState state =
                    generate_claim_state(DC_VALIDATION_SEED_START + (uint64_t)i);
                size_t action = rules[rule_index].action;
                size_t offset = i * DC_ACTION_COUNT + action;
                if (action != state.target && !covered[offset] &&
                    claim_rule_rejects_state(&rules[rule_index], &state, action)) {
                    ++marginal;
                }
            }
            if (marginal > best_marginal ||
                (marginal == best_marginal && best == rule_count)) {
                best = rule_index;
                best_marginal = marginal;
            }
        }
        if (best == rule_count || best_marginal == 0) {
            break;
        }
        selected[best] = 1;
        accepted[*accepted_count] = best;
        ++*accepted_count;
        for (size_t i = 0; i < DC_VALIDATION_SAMPLES; ++i) {
            ClaimState state = generate_claim_state(DC_VALIDATION_SEED_START + (uint64_t)i);
            size_t action = rules[best].action;
            if (action != state.target &&
                claim_rule_rejects_state(&rules[best], &state, action)) {
                covered[i * DC_ACTION_COUNT + action] = 1;
            }
        }
    }

    free(covered);
    free(selected);
    return 1;
}

static const char *claim_proposer_feature_set_name(ClaimProposerMode mode)
{
    switch (mode) {
    case CLAIM_PROPOSER_CLOSED_LOOP:
        return "claim_threshold_margin_residual_rows_v1";
    case CLAIM_PROPOSER_VALIDATION_ORACLE:
        return "claim_threshold_margin_validation_oracle";
    default:
        return "claim_threshold_margin_structural_rows_v1";
    }
}

static const char *claim_proposer_target_name(ClaimProposerMode mode)
{
    switch (mode) {
    case CLAIM_PROPOSER_CLOSED_LOOP:
        return "validation_marginal_support";
    case CLAIM_PROPOSER_VALIDATION_ORACLE:
        return "validation_support_oracle";
    case CLAIM_PROPOSER_NEURAL:
        return "train_support";
    default:
        return "none";
    }
}

static int emit_claim_compare_json(const char *model_hash,
                                   const char *parent_hash,
                                   const char *evaluation_mode,
                                   const char *gate_strategy,
                                   const ClaimMetrics *metrics,
                                   size_t accepted_rules,
                                   const ClaimEvalConfig *eval_config,
                                   ClaimRuleFamily rule_family)
{
    const ClaimEvalConfig *config =
        eval_config == NULL ? &DC_EVAL_CANONICAL : eval_config;
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"delayed_claim\","
           "\"evaluation_mode\":\"%s\","
           "\"gate_strategy\":\"%s\","
           "\"evaluation_band\":\"%s\","
           "\"rule_family\":\"%s\","
           "\"observation_noise\":%.8f,"
           "\"uncertainty_band\":%.8f,"
           "\"target_source\":\"hidden_clean_state\","
           "\"accepted_rules\":%zu,"
           "\"false_rejections\":%zu,"
           "\"support_fallback_margin\":%.8f,"
           "\"support_fallbacks\":%zu,"
           "\"leakage_policy\":\"%s\","
           "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
           "\"feature_encoder_version\":1,"
           "\"benchmark_seed\":%llu,\"sample_count\":%zu,"
           "\"accuracy\":%.8f,\"nll\":%.8f,\"ece\":%.8f,"
           "\"decision_utility\":%.8f,\"proposal_steps\":%.8f,"
           "\"baseline_steps\":%.8f,\"candidate_count\":%.8f,"
           "\"rejected_actions\":%.8f,\"search_saved\":%.8f",
           evaluation_mode,
           gate_strategy,
           config->label,
           accepted_rules == 0 ? "none" : claim_rule_family_name(rule_family),
           config->observation_noise,
           config->uncertainty_band,
           accepted_rules,
           metrics->false_rejections,
           config->support_fallback_margin,
           metrics->support_fallbacks,
           nn_leakage_policy_name(DELAYED_CLAIM_EXPERIMENT_SPEC.leakage_policy),
           model_hash == NULL ? "unknown" : model_hash,
           parent_hash == NULL || parent_hash[0] == '\0' ? "unknown" : parent_hash,
           (unsigned long long)config->seed_start,
           metrics->samples,
           metrics->samples == 0 ? 0.0 : (double)metrics->correct / (double)metrics->samples,
           metrics->nll,
           metrics->ece,
           metrics->utility,
           metrics->proposal_steps,
           metrics->baseline_steps,
           metrics->candidate_count,
           metrics->rejected_actions,
           metrics->baseline_steps - metrics->proposal_steps);
    print_claim_reliability_bins_json(metrics);
    printf("}\n");
    return 1;
}

static void emit_claim_robustness_json(const ClaimMetrics *metrics,
                                       const ClaimEvalConfig *config,
                                       const char *policy)
{
    double n = metrics->samples == 0 ? 1.0 : (double)metrics->samples;
    double p = (double)metrics->false_rejections / n;
    const double z = 1.959963984540054;
    double denominator = 1.0 + z * z / n;
    double center = (p + z * z / (2.0 * n)) / denominator;
    double half = z * sqrt((p * (1.0 - p) / n) + z * z / (4.0 * n * n)) /
                  denominator;
    printf("{\"schema\":\"crlp.delayed_claim.robustness.v1\","
           "\"task\":\"delayed_claim\",\"policy\":\"%s\","
           "\"observation_noise\":%.8f,\"support_fallback_margin\":%.8f,"
           "\"sample_count\":%zu,\"false_rejections\":%zu,"
           "\"false_rejection_rate\":%.10f,\"wilson95_low\":%.10f,"
           "\"wilson95_high\":%.10f,\"support_fallbacks\":%zu,"
           "\"support_fallback_rate\":%.10f,\"accuracy\":%.10f,"
           "\"decision_utility\":%.10f}\n",
           policy,
           config->observation_noise,
           config->support_fallback_margin,
           metrics->samples,
           metrics->false_rejections,
           p,
           fmax(0.0, center - half),
           fmin(1.0, center + half),
           metrics->support_fallbacks,
           (double)metrics->support_fallbacks / n,
           (double)metrics->correct / n,
           metrics->utility);
}

static void emit_claim_rule_summary_json(const ClaimRuleSearchConfig *config,
                                         const ClaimRule *rules,
                                         size_t rule_count,
                                         const size_t *accepted,
                                         const ClaimRuleGateMetrics *metrics)
{
    size_t verified = 0;
    for (size_t i = 0; i < rule_count; ++i) {
        verified += rules[i].certified ? 1u : 0u;
    }
    char rule_hash[32];
    snprintf(rule_hash, sizeof(rule_hash), "%016llx", (unsigned long long)metrics->rule_set_hash);
    char grammar_json[64];
    snprintf(grammar_json,
             sizeof(grammar_json),
             "\"%s\"",
             claim_rule_grammar_version(config->rule_family));
    char accepted_json[512];
    size_t written = 0;
    if (sizeof(accepted_json) != 0) {
        accepted_json[0] = '\0';
    }
    for (size_t i = 0; i < metrics->accepted_rules; ++i) {
        int added = snprintf(&accepted_json[written],
                             sizeof(accepted_json) - written,
                             "%s%zu",
                             i == 0 ? "" : ",",
                             accepted[i]);
        if (added < 0 || (size_t)added >= sizeof(accepted_json) - written) {
            accepted_json[sizeof(accepted_json) - 1u] = '\0';
            break;
        }
        written += (size_t)added;
    }
    char extra_json[1024];
    snprintf(extra_json,
             sizeof(extra_json),
             "\"rule_family\":\"%s\","
             "\"include_state_conditions\":true,"
             "\"include_candidate_filters\":false,"
             "\"include_gap_filters\":false,"
             "\"accepted_rule_ids\":[%s]",
             claim_rule_family_name(config->rule_family),
             accepted_json);
    NNRuleSearchSummary summary = {
        .report_version = 1,
        .task_name = "delayed_claim",
        .grammar_version_json = grammar_json,
        .rule_set_hash = rule_hash,
        .max_rules = config->max_rules,
        .selection_strategy = config->selection_strategy,
        .selection_seed = config->seed,
        .proposed_rules = rule_count,
        .verified_rules = verified,
        .accepted_rules = metrics->accepted_rules,
        .false_rejections = metrics->false_rejections,
        .proposer_feature_set = claim_proposer_feature_set_name(config->mode),
        .proposer_target = claim_proposer_target_name(config->mode),
        .train_marginal_samples = DC_VALIDATION_SAMPLES,
        .train_rejects = metrics->train_rejects,
        .validation_rejects = metrics->validation_rejects,
        .test_rejects = metrics->test_rejects,
        .direct_exact = metrics->direct_exact,
        .expanded_exact = metrics->expanded_exact,
        .expanded_saved_pct = metrics->expanded_saved_pct,
        .extra_json = extra_json,
    };
    nn_rule_search_summary_write_json(stdout, &summary);
}

static int claim_rule_accepted_rank(const size_t *accepted,
                                    size_t accepted_count,
                                    size_t rule_id,
                                    size_t *out_rank)
{
    for (size_t i = 0; i < accepted_count; ++i) {
        if (accepted[i] == rule_id) {
            *out_rank = i + 1u;
            return 1;
        }
    }
    *out_rank = 0u;
    return 0;
}

static const char *claim_rule_verifier_result(const ClaimRule *rule, int accepted)
{
    if (accepted) {
        return "accepted";
    }
    if (rule->train_false_rejections != 0u ||
        rule->validation_false_rejections != 0u) {
        return "unsafe";
    }
    if (rule->train_rejects == 0u && rule->validation_rejects == 0u) {
        return "no_support";
    }
    return rule->certified ? "verified_not_selected" : "unsafe";
}

static void emit_claim_rule_trace_json(FILE *trace,
                                       const ClaimRuleSearchConfig *config,
                                       const ClaimRule *rule,
                                       int accepted,
                                       size_t rank)
{
    if (trace == NULL) {
        return;
    }
    char extra_json[512];
    snprintf(extra_json,
             sizeof(extra_json),
             "\"action\":%zu,"
             "\"action_name\":\"%s\","
             "\"feature\":%zu,"
             "\"other_action\":%zu,"
             "\"direction\":%d,"
             "\"threshold\":%.12g,"
             "\"margin\":%.12g,"
             "\"rank\":%zu",
             rule->action,
             claim_action_name(rule->action),
             rule->feature,
             rule->other_action,
             rule->direction,
             rule->threshold,
             rule->margin,
             rank);
    NNRuleTraceRecord record = {
        .schema = "crlp.delayed_claim.rule_trace.v1",
        .task_name = "delayed_claim",
        .selection_strategy = config->selection_strategy,
        .rule_id = rule->id,
        .rule_rank = rank,
        .rule_kind = claim_rule_kind_name(rule->kind),
        .accepted = accepted,
        .verifier_result = claim_rule_verifier_result(rule, accepted),
        .certificate = "finite_train_validation_safety",
        .model_score = rule->neural_score,
        .structural_score = rule->structural_score,
        .train_rejects = rule->train_rejects,
        .train_false_rejections = rule->train_false_rejections,
        .validation_rejects = rule->validation_rejects,
        .validation_false_rejections = rule->validation_false_rejections,
        .test_rejects = rule->test_rejects,
        .test_false_rejections = rule->test_false_rejections,
        .train_marginal_utility = (double)rule->train_rejects,
        .validation_marginal_utility = (double)rule->validation_rejects,
        .extra_json = extra_json,
    };
    nn_rule_trace_write_json(trace, &record);
}

static int parse_claim_proposer_mode(const char *text, ClaimProposerMode *out_mode)
{
    if (strcmp(text, "neural") == 0 || strcmp(text, "neural_structural") == 0) {
        *out_mode = CLAIM_PROPOSER_NEURAL;
    } else if (strcmp(text, "closed-loop") == 0 ||
               strcmp(text, "closed_loop") == 0 ||
               strcmp(text, "closed_loop_neural") == 0) {
        *out_mode = CLAIM_PROPOSER_CLOSED_LOOP;
    } else if (strcmp(text, "random") == 0) {
        *out_mode = CLAIM_PROPOSER_RANDOM;
    } else if (strcmp(text, "grammar") == 0 || strcmp(text, "grammar_order") == 0) {
        *out_mode = CLAIM_PROPOSER_GRAMMAR;
    } else if (strcmp(text, "structural") == 0 ||
               strcmp(text, "structural_heuristic") == 0) {
        *out_mode = CLAIM_PROPOSER_STRUCTURAL;
    } else if (strcmp(text, "validation-oracle") == 0 ||
               strcmp(text, "validation_oracle") == 0) {
        *out_mode = CLAIM_PROPOSER_VALIDATION_ORACLE;
    } else {
        return 0;
    }
    return 1;
}

static const char *claim_proposer_strategy_name(ClaimProposerMode mode)
{
    switch (mode) {
    case CLAIM_PROPOSER_NEURAL:
        return "neural_structural";
    case CLAIM_PROPOSER_CLOSED_LOOP:
        return "closed_loop_neural";
    case CLAIM_PROPOSER_RANDOM:
        return "random";
    case CLAIM_PROPOSER_GRAMMAR:
        return "grammar_order";
    case CLAIM_PROPOSER_STRUCTURAL:
        return "structural_heuristic";
    case CLAIM_PROPOSER_VALIDATION_ORACLE:
        return "validation_oracle";
    }
    return "unknown";
}

static int run_claim_rule_search_for_mode(NeuralNet *net,
                                          const char *model_hash,
                                          const char *parent_hash,
                                          ClaimRule *rules,
                                          size_t rule_count,
                                          ClaimRuleSearchConfig config,
                                          nn_real *candidate_inputs,
                                          nn_real *scores,
                                          nn_real *probabilities,
                                          FILE *rule_trace)
{
    size_t *accepted = calloc(config.max_rules == 0 ? 1 : config.max_rules,
                              sizeof(*accepted));
    if (accepted == NULL) {
        return 0;
    }

    size_t accepted_count = 0;
    int ok = 0;
    if (config.mode == CLAIM_PROPOSER_CLOSED_LOOP) {
        ok = select_claim_rules_closed_loop(rules,
                                            rule_count,
                                            config.max_rules,
                                            accepted,
                                            &accepted_count);
    } else {
        ok = select_claim_rules_ordered(rules,
                                        rule_count,
                                        config.mode,
                                        config.seed,
                                        config.max_rules,
                                        accepted,
                                        &accepted_count);
    }
    if (!ok) {
        free(accepted);
        return 0;
    }

    ClaimRuleGateMetrics gate_metrics =
        evaluate_claim_rule_gate(rules, rule_count, accepted, accepted_count);
    emit_claim_rule_summary_json(&config, rules, rule_count, accepted, &gate_metrics);
    if (rule_trace != NULL) {
        for (size_t rule_index = 0; rule_index < rule_count; ++rule_index) {
            size_t rank = 0u;
            int is_accepted = claim_rule_accepted_rank(accepted,
                                                       accepted_count,
                                                       rules[rule_index].id,
                                                       &rank);
            emit_claim_rule_trace_json(rule_trace,
                                       &config,
                                       &rules[rule_index],
                                       is_accepted,
                                       rank);
        }
    }

    ClaimRule *selected_rules = NULL;
    if (accepted_count != 0) {
        selected_rules = calloc(accepted_count, sizeof(*selected_rules));
        if (selected_rules == NULL) {
            free(accepted);
            return 0;
        }
        for (size_t i = 0; i < accepted_count; ++i) {
            selected_rules[i] = rules[accepted[i]];
        }
    }

    char mode_name[96];
    const char *gate_prefix = "certified_threshold_margin_gate";
    if (config.rule_family == CLAIM_RULE_FAMILY_INDIRECT) {
        gate_prefix = "certified_threshold_risk_margin_gate";
    } else if (config.rule_family == CLAIM_RULE_FAMILY_ALL) {
        gate_prefix = "certified_threshold_mixed_margin_gate";
    }
    snprintf(mode_name,
             sizeof(mode_name),
             "%s_%s",
             gate_prefix,
             config.selection_strategy);
    ClaimMetrics model_metrics = evaluate_claim_model_with_gate_on_band(net,
                                                                        &DC_EVAL_CANONICAL,
                                                                        selected_rules,
                                                                        accepted_count,
                                                                        candidate_inputs,
                                                                        scores,
                                                                        probabilities);
    emit_claim_compare_json(model_hash,
                            parent_hash,
                            mode_name,
                            config.selection_strategy,
                            &model_metrics,
                            accepted_count,
                            &DC_EVAL_CANONICAL,
                            config.rule_family);
    print_claim_metrics(mode_name, model_hash, model_metrics);

    for (size_t band_index = 0;
         band_index < sizeof(DC_EVAL_STRESS_BANDS) / sizeof(DC_EVAL_STRESS_BANDS[0]);
         ++band_index) {
        const ClaimEvalConfig *band = &DC_EVAL_STRESS_BANDS[band_index];
        char stress_mode_name[128];
        snprintf(stress_mode_name,
                 sizeof(stress_mode_name),
                 "%s_%s",
                 mode_name,
                 band->label);
        ClaimMetrics stress_metrics =
            evaluate_claim_model_with_gate_on_band(net,
                                                   band,
                                                   selected_rules,
                                                   accepted_count,
                                                   candidate_inputs,
                                                   scores,
                                                   probabilities);
        emit_claim_compare_json(model_hash,
                                parent_hash,
                                stress_mode_name,
                                config.selection_strategy,
                                &stress_metrics,
                                accepted_count,
                                band,
                                config.rule_family);
        print_claim_metrics(stress_mode_name, model_hash, stress_metrics);
    }

    if (config.mode == CLAIM_PROPOSER_NEURAL &&
        config.rule_family == CLAIM_RULE_FAMILY_DIRECT) {
        static const double noise_levels[] = {0.0, 0.01, 0.02, 0.03, 0.05, 0.08};
        for (size_t noise_index = 0;
             noise_index < sizeof(noise_levels) / sizeof(noise_levels[0]);
             ++noise_index) {
            for (size_t policy_index = 0; policy_index < 2; ++policy_index) {
                ClaimEvalConfig robustness = {
                    "robustness_sweep",
                    DC_TEST_SEED_START,
                    DC_TEST_SAMPLES,
                    noise_levels[noise_index],
                    0.0,
                    DC_TEST_SAMPLES,
                    policy_index == 0 ? 0.0 : 0.08,
                };
                ClaimMetrics robustness_metrics =
                    evaluate_claim_model_with_gate_on_band(net,
                                                           &robustness,
                                                           selected_rules,
                                                           accepted_count,
                                                           candidate_inputs,
                                                           scores,
                                                           probabilities);
                emit_claim_robustness_json(&robustness_metrics,
                                           &robustness,
                                           policy_index == 0 ? "direct_gate"
                                                             : "support_aware_fallback");
            }
        }
    }
    free(selected_rules);
    free(accepted);
    return 1;
}

static int run_claim_rule_search(const char *model_path,
                                 size_t max_rules,
                                 const char *proposer_text,
                                 uint64_t proposer_seed,
                                 ClaimRuleFamily rule_family,
                                 const char *rule_trace_path)
{
    NNCheckpointMeta meta;
    NeuralNet *net = nn_load_checkpoint(model_path, &meta);
    if (net == NULL) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 0;
    }
    char hash[65] = "unknown";
    nn_hash_checkpoint_hex(model_path, hash);

    ClaimRule *rules = NULL;
    size_t rule_count = enumerate_claim_rule_candidates(&rules, rule_family);
    if (rule_count == 0 || rules == NULL) {
        nn_free(net);
        return 0;
    }
    audit_claim_rules(rules, rule_count);
    score_claim_rules_with_neural_prior(rules, rule_count);

    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(DC_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(DC_ACTION_COUNT, sizeof(*probabilities));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL) {
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        free(rules);
        nn_free(net);
        return 0;
    }

    FILE *rule_trace = NULL;
    if (rule_trace_path != NULL) {
        rule_trace = fopen(rule_trace_path, "w");
        if (rule_trace == NULL) {
            fprintf(stderr, "failed to open rule trace: %s\n", rule_trace_path);
            free(candidate_inputs);
            free(scores);
            free(probabilities);
            free(rules);
            nn_free(net);
            return 0;
        }
    }

    ClaimProposerMode modes[] = {
        CLAIM_PROPOSER_NEURAL,
        CLAIM_PROPOSER_CLOSED_LOOP,
        CLAIM_PROPOSER_RANDOM,
        CLAIM_PROPOSER_GRAMMAR,
        CLAIM_PROPOSER_STRUCTURAL,
        CLAIM_PROPOSER_VALIDATION_ORACLE,
    };
    size_t mode_count = sizeof(modes) / sizeof(modes[0]);
    if (strcmp(proposer_text, "all") != 0) {
        if (!parse_claim_proposer_mode(proposer_text, &modes[0])) {
            fprintf(stderr, "unknown delayed_claim proposer: %s\n", proposer_text);
            if (rule_trace != NULL) {
                fclose(rule_trace);
            }
            free(candidate_inputs);
            free(scores);
            free(probabilities);
            free(rules);
            nn_free(net);
            return 0;
        }
        mode_count = 1;
    }

    int ok = 1;
    for (size_t i = 0; i < mode_count; ++i) {
        ClaimRuleSearchConfig config;
        config.mode = modes[i];
        config.label = proposer_text;
        config.selection_strategy = claim_proposer_strategy_name(modes[i]);
        config.rule_family = rule_family;
        config.max_rules = max_rules;
        config.seed = proposer_seed;
        ok = ok && run_claim_rule_search_for_mode(net,
                                                  hash,
                                                  meta.parent_model_hash,
                                                  rules,
                                                  rule_count,
                                                  config,
                                                  candidate_inputs,
                                                  scores,
                                                  probabilities,
                                                  rule_trace);
    }

    if (rule_trace != NULL && fclose(rule_trace) != 0) {
        ok = 0;
    }
    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free(rules);
    nn_free(net);
    return ok;
}

static void print_usage(FILE *stream);

static int compare_claim_models(int argc, char **argv)
{
    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(DC_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(DC_ACTION_COUNT, sizeof(*probabilities));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL) {
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        return 0;
    }

    int model_start = 2;
    if (argc >= 4 && strcmp(argv[2], "--test-seed") == 0) {
        if (!nn_parse_u64_text(argv[3], &dc_eval_test_seed_start)) {
            print_usage(stderr);
            free(candidate_inputs);
            free(scores);
            free(probabilities);
            return 0;
        }
        model_start = 4;
    }
    if (model_start >= argc) {
        print_usage(stderr);
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        return 0;
    }

    int ok = 1;
    for (int i = model_start; i < argc; ++i) {
        NNCheckpointMeta meta;
        NeuralNet *net = nn_load_checkpoint(argv[i], &meta);
        char hash[65] = "unknown";
        if (net == NULL) {
            fprintf(stderr, "failed to load model: %s\n", argv[i]);
            ok = 0;
            continue;
        }
        nn_hash_checkpoint_hex(argv[i], hash);
        ClaimMetrics metrics = evaluate_claim_model(net,
                                                    dc_eval_test_seed_start,
                                                    DC_TEST_SAMPLES,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
        printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"delayed_claim\","
               "\"evaluation_mode\":\"raw\","
               "\"gate_strategy\":\"none\","
               "\"evaluation_band\":\"canonical\","
               "\"rule_family\":\"none\","
               "\"observation_noise\":0.00000000,"
               "\"uncertainty_band\":0.00000000,"
               "\"target_source\":\"hidden_clean_state\","
               "\"accepted_rules\":0,"
               "\"false_rejections\":%zu,"
               "\"leakage_policy\":\"%s\","
               "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
               "\"feature_encoder_version\":%u,"
               "\"benchmark_seed\":%u,\"sample_count\":%zu,"
               "\"accuracy\":%.8f,\"nll\":%.8f,\"ece\":%.8f,"
               "\"decision_utility\":%.8f,\"proposal_steps\":%.8f,"
               "\"baseline_steps\":%.8f,\"candidate_count\":%.8f,"
               "\"rejected_actions\":%.8f,\"search_saved\":%.8f",
               metrics.false_rejections,
               nn_leakage_policy_name(DELAYED_CLAIM_EXPERIMENT_SPEC.leakage_policy),
               hash,
               meta.parent_model_hash[0] == '\0' ? "unknown" : meta.parent_model_hash,
               meta.feature_encoder_version,
               (unsigned int)dc_eval_test_seed_start,
               metrics.samples,
               metrics.samples == 0 ? 0.0 : (double)metrics.correct / (double)metrics.samples,
               metrics.nll,
               metrics.ece,
               metrics.utility,
               metrics.proposal_steps,
               metrics.baseline_steps,
               metrics.candidate_count,
               metrics.rejected_actions,
               metrics.baseline_steps - metrics.proposal_steps);
        print_claim_reliability_bins_json(&metrics);
        printf("}\n");
        print_claim_metrics(argv[i], hash, metrics);
        nn_free(net);
    }

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    return ok;
}

static void print_usage(FILE *stream)
{
    fputs("usage: delayed_claim [--trace path] [--save path] "
          "[--trace-train] [--trace-seed seed] [--trace-samples count]\n"
          "       delayed_claim --replay trace.jsonl --out model.nnckpt "
          "[--epochs count] [--hard-weight value] [--soft-weight value]\n"
          "       delayed_claim --compare [--test-seed seed] model_a.nnckpt model_b.nnckpt [...]\n"
          "       delayed_claim --rule-search model.nnckpt "
          "[--max-rules count] [--rule-family direct|indirect|all] "
          "[--proposer all|neural|closed-loop|random|grammar|structural|validation-oracle] "
          "[--rule-trace path]\n"
          "       delayed_claim --schema\n",
          stream);
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--schema") == 0) {
        return nn_experiment_write_schema_json(stdout, &DELAYED_CLAIM_EXPERIMENT_SPEC)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--self-test-reordered-trace") == 0) {
        return delayed_claim_reordered_trace_self_test() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--replay") == 0) {
        if (argc < 5) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }

        ClaimReplayConfig replay_config;
        replay_config.trace_path = argv[2];
        replay_config.out_path = NULL;
        replay_config.epochs = DC_REPLAY_EPOCHS;
        replay_config.learning_rate = DC_REPLAY_LEARNING_RATE;
        replay_config.hard_weight = DC_REPLAY_HARD_WEIGHT;
        replay_config.soft_weight = DC_REPLAY_SOFT_WEIGHT;

        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                replay_config.out_path = argv[++i];
            } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
                if (!nn_parse_size_text(argv[++i], &replay_config.epochs)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
                if (!nn_parse_real_nonnegative_text(argv[++i], &replay_config.learning_rate) ||
                    replay_config.learning_rate <= (nn_real)0) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--hard-weight") == 0 && i + 1 < argc) {
                if (!nn_parse_real_nonnegative_text(argv[++i], &replay_config.hard_weight)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--soft-weight") == 0 && i + 1 < argc) {
                if (!nn_parse_real_nonnegative_text(argv[++i], &replay_config.soft_weight)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        }

        if (replay_config.out_path == NULL ||
            (replay_config.hard_weight == (nn_real)0 &&
             replay_config.soft_weight == (nn_real)0)) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        return replay_claim_trace(&replay_config) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--compare") == 0) {
        if (argc < 4) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        return compare_claim_models(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--rule-search") == 0) {
        if (argc < 3) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        const char *model_path = argv[2];
        const char *proposer = "all";
        const char *rule_trace_path = NULL;
        ClaimRuleFamily rule_family = CLAIM_RULE_FAMILY_DIRECT;
        size_t max_rules = 8;
        uint64_t proposer_seed = UINT64_C(20240511);
        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--max-rules") == 0 && i + 1 < argc) {
                if (!nn_parse_size_text(argv[++i], &max_rules)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--proposer") == 0 && i + 1 < argc) {
                proposer = argv[++i];
            } else if (strcmp(argv[i], "--rule-family") == 0 && i + 1 < argc) {
                if (!parse_claim_rule_family(argv[++i], &rule_family)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--proposer-seed") == 0 && i + 1 < argc) {
                if (!nn_parse_u64_text(argv[++i], &proposer_seed)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--rule-trace") == 0 && i + 1 < argc) {
                rule_trace_path = argv[++i];
            } else {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        }
        return run_claim_rule_search(model_path,
                                     max_rules,
                                     proposer,
                                     proposer_seed,
                                     rule_family,
                                     rule_trace_path)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }

    const char *trace_path = NULL;
    const char *save_path = NULL;
    uint64_t trace_seed_start = DC_TEST_SEED_START;
    size_t trace_samples = DC_TEST_SAMPLES;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--trace-train") == 0) {
            trace_seed_start = DC_TRAIN_SEED_START;
            trace_samples = DC_TRAIN_SAMPLES;
        } else if (strcmp(argv[i], "--trace-seed") == 0 && i + 1 < argc) {
            if (!nn_parse_u64_text(argv[++i], &trace_seed_start)) {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--trace-samples") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &trace_samples)) {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        } else {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
    }

    NeuralNet *net = create_claim_model(5150);
    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(DC_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(DC_ACTION_COUNT, sizeof(*probabilities));
    if (net == NULL || candidate_inputs == NULL || scores == NULL || probabilities == NULL) {
        fputs("failed to allocate delayed_claim model\n", stderr);
        nn_free(net);
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        return EXIT_FAILURE;
    }

    int ok = train_claim_model(net, DC_TRAIN_SEED_START, DC_TRAIN_SAMPLES, 4, candidate_inputs);
    ClaimMetrics metrics = evaluate_claim_model(net,
                                                DC_TEST_SEED_START,
                                                DC_TEST_SAMPLES,
                                                candidate_inputs,
                                                scores,
                                                probabilities);
    char hash[65] = "unknown";
    if (ok && save_path != NULL) {
        ok = save_claim_checkpoint(net,
                                   save_path,
                                   "unknown",
                                   "delayed_claim trained checkpoint",
                                   hash);
        if (ok) {
            printf("saved delayed_claim checkpoint: %s hash=%s\n", save_path, hash);
        }
    }
    print_claim_metrics("delayed_claim", hash, metrics);

    if (ok && trace_path != NULL) {
        ok = emit_claim_trace(net,
                              trace_path,
                              hash,
                              trace_seed_start,
                              trace_samples,
                              candidate_inputs,
                              scores,
                              probabilities);
    }

    nn_free(net);
    free(candidate_inputs);
    free(scores);
    free(probabilities);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
