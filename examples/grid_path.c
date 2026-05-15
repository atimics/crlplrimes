#include "nn.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GP_GRID = 7,
    GP_CELLS = GP_GRID * GP_GRID,
    GP_ACTION_COUNT = 4,
    GP_FEATURE_ENCODER_VERSION = 3,
    GP_RULE_SEARCH_GRAMMAR_VERSION = 2,
    GP_RULE_FEATURE_COUNT = 10,
    GP_RULE_COUNT = 5,
    GP_RULE_MAX_ACCEPTED = 3,
    GP_RULE_SCORER_EPOCHS = 96,
    GP_MAP_FEATURE_OFFSET = 10,
    GP_ACTION_FEATURE_OFFSET = GP_MAP_FEATURE_OFFSET + GP_CELLS,
    GP_FEATURE_COUNT = GP_ACTION_FEATURE_OFFSET + GP_ACTION_COUNT,
    GP_TRAIN_SAMPLES = 4000,
    GP_VALIDATION_SAMPLES = 1000,
    GP_TEST_SAMPLES = 1000,
    GP_TRAIN_SEED_START = 7000,
    GP_VALIDATION_SEED_START = 450000,
    GP_TEST_SEED_START = 900000,
    GP_REPLAY_EPOCHS = 5,
    GP_CALIBRATION_BINS = 10,
    GP_INF = 1000000,
};

typedef struct {
    uint8_t blocked[GP_CELLS];
    int sx;
    int sy;
    int gx;
    int gy;
} GridState;

typedef struct {
    size_t samples;
    size_t correct;
    double candidate_count;
    double rejected_candidates;
    double nll;
    double ece;
    double utility;
    double proposal_steps;
    double baseline_steps;
    double verifier_nodes;
    double baseline_nodes;
    size_t bin_count[GP_CALIBRATION_BINS];
    double bin_confidence[GP_CALIBRATION_BINS];
    double bin_accuracy[GP_CALIBRATION_BINS];
} GridMetrics;

typedef struct {
    uint64_t *seeds;
    uint32_t *target_action_ids;
    size_t count;
    size_t capacity;
    char parent_hash[65];
} GridTrace;

typedef enum {
    GP_RULE_UP_AT_TOP = 0,
    GP_RULE_RIGHT_AT_RIGHT_EDGE = 1,
    GP_RULE_DOWN_AT_BOTTOM = 2,
    GP_RULE_LEFT_AT_LEFT_EDGE = 3,
    GP_RULE_BLOCKED_NEXT = 4,
} GridGateRuleKind;

typedef struct {
    GridGateRuleKind kind;
    size_t train_rejects;
    size_t validation_rejects;
    size_t test_rejects;
    size_t train_false_rejections;
    size_t validation_false_rejections;
    size_t test_false_rejections;
    nn_real model_score;
    int accepted;
} GridGateRule;

typedef struct {
    GridGateRule items[GP_RULE_COUNT];
    size_t count;
    size_t accepted_rules;
    size_t train_rejects;
    size_t validation_rejects;
    size_t test_rejects;
    size_t train_false_rejections;
    size_t validation_false_rejections;
    size_t test_false_rejections;
    double proposer_loss;
} GridGateRuleSet;

static const int ACTION_DX[GP_ACTION_COUNT] = {0, 1, 0, -1};
static const int ACTION_DY[GP_ACTION_COUNT] = {-1, 0, 1, 0};

static uint32_t gp_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double gp_rng_unit(uint32_t *state)
{
    return (double)gp_rng_next(state) / (double)UINT32_MAX;
}

static int parse_size_text(const char *text, size_t *out_value)
{
    if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || value == 0) {
        return 0;
    }

    *out_value = (size_t)value;
    return 1;
}

static int parse_u64_text(const char *text, uint64_t *out_value)
{
    if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) {
        return 0;
    }

    *out_value = (uint64_t)value;
    return 1;
}

static int cell_index(int x, int y)
{
    return y * GP_GRID + x;
}

static int in_bounds(int x, int y)
{
    return x >= 0 && x < GP_GRID && y >= 0 && y < GP_GRID;
}

static double safe_probability(double p)
{
    return p < 1e-12 ? 1e-12 : p;
}

static uint32_t grid_action_id_for_action(size_t action)
{
    return (uint32_t)action;
}

static int grid_action_from_action_id(uint32_t action_id, size_t *out_action)
{
    if (out_action == NULL || action_id >= GP_ACTION_COUNT) {
        return 0;
    }
    *out_action = (size_t)action_id;
    return 1;
}

static size_t grid_row_for_action_id(const uint32_t *action_ids,
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

static size_t bfs_distances_from_goal(const GridState *state, int *dist)
{
    int queue[GP_CELLS];
    size_t head = 0;
    size_t tail = 0;
    size_t expansions = 0;

    for (size_t i = 0; i < GP_CELLS; ++i) {
        dist[i] = GP_INF;
    }

    int goal = cell_index(state->gx, state->gy);
    dist[goal] = 0;
    queue[tail++] = goal;

    while (head < tail) {
        int cell = queue[head++];
        int x = cell % GP_GRID;
        int y = cell / GP_GRID;
        ++expansions;

        for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
            int nx = x + ACTION_DX[action];
            int ny = y + ACTION_DY[action];
            if (!in_bounds(nx, ny)) {
                continue;
            }
            int next = cell_index(nx, ny);
            if (state->blocked[next] || dist[next] != GP_INF) {
                continue;
            }
            dist[next] = dist[cell] + 1;
            queue[tail++] = next;
        }
    }

    return expansions;
}

static size_t bfs_nodes_from_cell(const GridState *state, int sx, int sy)
{
    uint8_t seen[GP_CELLS] = {0};
    int queue[GP_CELLS];
    size_t head = 0;
    size_t tail = 0;
    size_t expansions = 0;

    if (!in_bounds(sx, sy) || state->blocked[cell_index(sx, sy)]) {
        return 1;
    }

    int start = cell_index(sx, sy);
    int goal = cell_index(state->gx, state->gy);
    seen[start] = 1;
    queue[tail++] = start;

    while (head < tail) {
        int cell = queue[head++];
        int x = cell % GP_GRID;
        int y = cell / GP_GRID;
        ++expansions;
        if (cell == goal) {
            return expansions;
        }

        for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
            int nx = x + ACTION_DX[action];
            int ny = y + ACTION_DY[action];
            if (!in_bounds(nx, ny)) {
                continue;
            }
            int next = cell_index(nx, ny);
            if (state->blocked[next] || seen[next]) {
                continue;
            }
            seen[next] = 1;
            queue[tail++] = next;
        }
    }

    return expansions == 0 ? 1 : expansions;
}

static int generate_grid_state(uint64_t seed, GridState *state)
{
    for (uint64_t attempt = 0; attempt < 1000; ++attempt) {
        uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ (attempt * 0x9e3779b9u));
        memset(state, 0, sizeof(*state));

        state->sx = (int)(gp_rng_next(&rng) % GP_GRID);
        state->sy = (int)(gp_rng_next(&rng) % GP_GRID);
        do {
            state->gx = (int)(gp_rng_next(&rng) % GP_GRID);
            state->gy = (int)(gp_rng_next(&rng) % GP_GRID);
        } while (state->gx == state->sx && state->gy == state->sy);

        for (int y = 0; y < GP_GRID; ++y) {
            for (int x = 0; x < GP_GRID; ++x) {
                int cell = cell_index(x, y);
                if ((x == state->sx && y == state->sy) ||
                    (x == state->gx && y == state->gy)) {
                    continue;
                }
                state->blocked[cell] = gp_rng_unit(&rng) < 0.18 ? 1u : 0u;
            }
        }

        int dist[GP_CELLS];
        bfs_distances_from_goal(state, dist);
        int start = cell_index(state->sx, state->sy);
        if (dist[start] > 0 && dist[start] < GP_INF) {
            return 1;
        }
    }
    return 0;
}

static size_t fill_grid_candidates(const GridState *state,
                                   nn_real *candidate_inputs,
                                   uint32_t *action_ids,
                                   size_t *target_index)
{
    int dist[GP_CELLS];
    bfs_distances_from_goal(state, dist);
    int best_dist = GP_INF;
    int manhattan_before = abs(state->gx - state->sx) + abs(state->gy - state->sy);

    *target_index = GP_ACTION_COUNT;
    for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
        int nx = state->sx + ACTION_DX[action];
        int ny = state->sy + ACTION_DY[action];
        int out_of_bounds = !in_bounds(nx, ny);
        int blocked_next = 0;
        int next_dist = GP_INF;
        int feature_x = state->sx;
        int feature_y = state->sy;

        if (!out_of_bounds) {
            int next_cell = cell_index(nx, ny);
            blocked_next = state->blocked[next_cell] ? 1 : 0;
            feature_x = nx;
            feature_y = ny;
            if (!blocked_next) {
                next_dist = dist[next_cell];
            }
        }

        int manhattan_after = abs(state->gx - feature_x) + abs(state->gy - feature_y);
        nn_real *row = &candidate_inputs[action * GP_FEATURE_COUNT];
        memset(row, 0, GP_FEATURE_COUNT * sizeof(*row));

        row[0] = (nn_real)((double)state->sx / (double)(GP_GRID - 1));
        row[1] = (nn_real)((double)state->sy / (double)(GP_GRID - 1));
        row[2] = (nn_real)((double)state->gx / (double)(GP_GRID - 1));
        row[3] = (nn_real)((double)state->gy / (double)(GP_GRID - 1));
        row[4] = (nn_real)((double)(state->gx - feature_x) / (double)(GP_GRID - 1));
        row[5] = (nn_real)((double)(state->gy - feature_y) / (double)(GP_GRID - 1));
        row[6] = (nn_real)((double)manhattan_after / (double)(2 * (GP_GRID - 1)));
        row[7] = (nn_real)((double)(manhattan_before - manhattan_after) / 2.0);
        row[8] = (nn_real)ACTION_DX[action];
        row[9] = (nn_real)ACTION_DY[action];
        for (size_t cell = 0; cell < GP_CELLS; ++cell) {
            row[GP_MAP_FEATURE_OFFSET + cell] = state->blocked[cell] ? (nn_real)1 : (nn_real)0;
        }
        for (size_t i = 0; i < GP_ACTION_COUNT; ++i) {
            row[GP_ACTION_FEATURE_OFFSET + i] = action == i ? (nn_real)1 : (nn_real)0;
        }

        action_ids[action] = grid_action_id_for_action(action);
        if (next_dist < GP_INF &&
            (next_dist < best_dist ||
             (next_dist == best_dist && *target_index == GP_ACTION_COUNT))) {
            best_dist = next_dist;
            *target_index = action;
        }
    }

    return GP_ACTION_COUNT;
}

static const char *grid_rule_name(GridGateRuleKind kind)
{
    switch (kind) {
    case GP_RULE_UP_AT_TOP:
        return "reject up when y == 0";
    case GP_RULE_RIGHT_AT_RIGHT_EDGE:
        return "reject right when x == max";
    case GP_RULE_DOWN_AT_BOTTOM:
        return "reject down when y == max";
    case GP_RULE_LEFT_AT_LEFT_EDGE:
        return "reject left when x == 0";
    case GP_RULE_BLOCKED_NEXT:
        return "reject action when next cell is blocked";
    }
    return "unknown";
}

static void init_grid_gate_rules(GridGateRuleSet *rules)
{
    memset(rules, 0, sizeof(*rules));
    rules->count = GP_RULE_COUNT;
    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].kind = (GridGateRuleKind)i;
    }
}

static int grid_gate_rule_has_certificate(const GridGateRule *rule)
{
    return rule != NULL && rule->kind >= GP_RULE_UP_AT_TOP &&
           rule->kind <= GP_RULE_BLOCKED_NEXT;
}

static int grid_gate_rule_rejects(const GridGateRule *rule,
                                  const GridState *state,
                                  size_t action)
{
    if (rule == NULL || state == NULL || action >= GP_ACTION_COUNT) {
        return 0;
    }

    int nx = state->sx + ACTION_DX[action];
    int ny = state->sy + ACTION_DY[action];
    switch (rule->kind) {
    case GP_RULE_UP_AT_TOP:
        return action == 0 && state->sy == 0;
    case GP_RULE_RIGHT_AT_RIGHT_EDGE:
        return action == 1 && state->sx == GP_GRID - 1;
    case GP_RULE_DOWN_AT_BOTTOM:
        return action == 2 && state->sy == GP_GRID - 1;
    case GP_RULE_LEFT_AT_LEFT_EDGE:
        return action == 3 && state->sx == 0;
    case GP_RULE_BLOCKED_NEXT:
        return in_bounds(nx, ny) && state->blocked[cell_index(nx, ny)];
    }
    return 0;
}

static int grid_gate_rules_reject_action(const GridGateRuleSet *rules,
                                         const GridState *state,
                                         size_t action)
{
    if (rules == NULL) {
        return 0;
    }
    for (size_t i = 0; i < rules->count; ++i) {
        const GridGateRule *rule = &rules->items[i];
        if (rule->accepted && grid_gate_rule_rejects(rule, state, action)) {
            return 1;
        }
    }
    return 0;
}

static size_t fill_grid_candidates_gated(const GridState *state,
                                         const GridGateRuleSet *rules,
                                         nn_real *candidate_inputs,
                                         uint32_t *action_ids,
                                         size_t *target_index)
{
    nn_real raw_inputs[(size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT];
    uint32_t raw_action_ids[GP_ACTION_COUNT];
    size_t raw_target = GP_ACTION_COUNT;
    fill_grid_candidates(state, raw_inputs, raw_action_ids, &raw_target);

    *target_index = GP_ACTION_COUNT;
    size_t candidate_count = 0;
    for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
        if (grid_gate_rules_reject_action(rules, state, action)) {
            continue;
        }
        memcpy(&candidate_inputs[candidate_count * GP_FEATURE_COUNT],
               &raw_inputs[action * GP_FEATURE_COUNT],
               GP_FEATURE_COUNT * sizeof(*candidate_inputs));
        action_ids[candidate_count] = raw_action_ids[action];
        if (action == raw_target) {
            *target_index = candidate_count;
        }
        ++candidate_count;
    }
    return candidate_count;
}

typedef enum {
    GP_RULE_SPLIT_TRAIN = 0,
    GP_RULE_SPLIT_VALIDATION = 1,
    GP_RULE_SPLIT_TEST = 2,
} GridRuleSplit;

static void increment_grid_rule_stats(GridGateRuleSet *rules,
                                      size_t rule_index,
                                      int false_rejection,
                                      GridRuleSplit split)
{
    GridGateRule *rule = &rules->items[rule_index];
    if (split == GP_RULE_SPLIT_TRAIN) {
        if (false_rejection) {
            ++rule->train_false_rejections;
        } else {
            ++rule->train_rejects;
        }
    } else if (split == GP_RULE_SPLIT_VALIDATION) {
        if (false_rejection) {
            ++rule->validation_false_rejections;
        } else {
            ++rule->validation_rejects;
        }
    } else {
        if (false_rejection) {
            ++rule->test_false_rejections;
        } else {
            ++rule->test_rejects;
        }
    }
}

static void accumulate_grid_rule_stats(GridGateRuleSet *rules,
                                       uint64_t seed_start,
                                       size_t samples,
                                       GridRuleSplit split)
{
    for (size_t i = 0; i < samples; ++i) {
        GridState state;
        size_t target_index = GP_ACTION_COUNT;
        nn_real candidate_inputs[(size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT];
        uint32_t action_ids[GP_ACTION_COUNT];
        if (!generate_grid_state(seed_start + (uint64_t)i, &state)) {
            continue;
        }
        fill_grid_candidates(&state, candidate_inputs, action_ids, &target_index);

        for (size_t rule_index = 0; rule_index < rules->count; ++rule_index) {
            GridGateRule *rule = &rules->items[rule_index];
            for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
                if (!grid_gate_rule_rejects(rule, &state, action)) {
                    continue;
                }
                increment_grid_rule_stats(rules,
                                          rule_index,
                                          action == target_index,
                                          split);
            }
        }
    }
}

static void fill_grid_rule_features(const GridGateRule *rule, nn_real *features)
{
    memset(features, 0, GP_RULE_FEATURE_COUNT * sizeof(*features));
    size_t action = 0;
    int uses_x = 0;
    int uses_y = 0;
    int low_edge = 0;
    int high_edge = 0;
    int observes_map = 0;

    switch (rule->kind) {
    case GP_RULE_UP_AT_TOP:
        action = 0;
        uses_y = 1;
        low_edge = 1;
        break;
    case GP_RULE_RIGHT_AT_RIGHT_EDGE:
        action = 1;
        uses_x = 1;
        high_edge = 1;
        break;
    case GP_RULE_DOWN_AT_BOTTOM:
        action = 2;
        uses_y = 1;
        high_edge = 1;
        break;
    case GP_RULE_LEFT_AT_LEFT_EDGE:
        action = 3;
        uses_x = 1;
        low_edge = 1;
        break;
    case GP_RULE_BLOCKED_NEXT:
        action = 0;
        observes_map = 1;
        break;
    }

    features[0] = (nn_real)((double)rule->kind / (double)(GP_RULE_COUNT - 1));
    features[1] = (nn_real)((double)action / (double)(GP_ACTION_COUNT - 1));
    features[2] = (nn_real)ACTION_DX[action];
    features[3] = (nn_real)ACTION_DY[action];
    features[4] = uses_x ? (nn_real)1 : (nn_real)0;
    features[5] = uses_y ? (nn_real)1 : (nn_real)0;
    features[6] = low_edge ? (nn_real)1 : (nn_real)0;
    features[7] = high_edge ? (nn_real)1 : (nn_real)0;
    features[8] = observes_map ? (nn_real)1 : (nn_real)0;
    features[9] = grid_gate_rule_has_certificate(rule) ? (nn_real)1 : (nn_real)0;
}

static int compare_grid_gate_rules(const void *left, const void *right)
{
    const GridGateRule *a = (const GridGateRule *)left;
    const GridGateRule *b = (const GridGateRule *)right;
    if (a->model_score > b->model_score) {
        return -1;
    }
    if (a->model_score < b->model_score) {
        return 1;
    }
    return (int)a->kind - (int)b->kind;
}

static int score_grid_gate_rules(GridGateRuleSet *rules)
{
    nn_real inputs[(size_t)GP_RULE_COUNT * GP_RULE_FEATURE_COUNT];
    nn_real targets[GP_RULE_COUNT];
    nn_real outputs[GP_RULE_COUNT];
    size_t max_rejects = 1;

    for (size_t i = 0; i < rules->count; ++i) {
        if (rules->items[i].train_rejects > max_rejects) {
            max_rejects = rules->items[i].train_rejects;
        }
    }
    for (size_t i = 0; i < rules->count; ++i) {
        fill_grid_rule_features(&rules->items[i], &inputs[i * GP_RULE_FEATURE_COUNT]);
        targets[i] = (nn_real)((double)rules->items[i].train_rejects /
                               (double)max_rejects);
    }

    const size_t layers[] = {GP_RULE_FEATURE_COUNT, 8, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 31415;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    rules->proposer_loss = (double)nn_train(net,
                                            inputs,
                                            targets,
                                            rules->count,
                                            (nn_real)0.05,
                                            GP_RULE_SCORER_EPOCHS,
                                            0);
    nn_forward_batch(net, inputs, outputs, rules->count);
    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].model_score = outputs[i];
    }
    nn_free(net);
    return 1;
}

static void summarize_grid_gate_rules(GridGateRuleSet *rules)
{
    rules->accepted_rules = 0;
    rules->train_rejects = 0;
    rules->validation_rejects = 0;
    rules->test_rejects = 0;
    rules->train_false_rejections = 0;
    rules->validation_false_rejections = 0;
    rules->test_false_rejections = 0;

    for (size_t i = 0; i < rules->count; ++i) {
        const GridGateRule *rule = &rules->items[i];
        if (!rule->accepted) {
            continue;
        }
        ++rules->accepted_rules;
        rules->train_rejects += rule->train_rejects;
        rules->validation_rejects += rule->validation_rejects;
        rules->test_rejects += rule->test_rejects;
        rules->train_false_rejections += rule->train_false_rejections;
        rules->validation_false_rejections += rule->validation_false_rejections;
        rules->test_false_rejections += rule->test_false_rejections;
    }
}

static int discover_grid_gate_rules(GridGateRuleSet *rules)
{
    init_grid_gate_rules(rules);
    accumulate_grid_rule_stats(rules,
                               GP_TRAIN_SEED_START,
                               GP_TRAIN_SAMPLES,
                               GP_RULE_SPLIT_TRAIN);
    accumulate_grid_rule_stats(rules,
                               GP_VALIDATION_SEED_START,
                               GP_VALIDATION_SAMPLES,
                               GP_RULE_SPLIT_VALIDATION);
    if (!score_grid_gate_rules(rules)) {
        return 0;
    }
    qsort(rules->items, rules->count, sizeof(*rules->items), compare_grid_gate_rules);

    for (size_t i = 0; i < rules->count; ++i) {
        GridGateRule *rule = &rules->items[i];
        rule->accepted = grid_gate_rule_has_certificate(rule) &&
                         rule->train_rejects != 0 &&
                         rule->train_false_rejections == 0 &&
                         rule->validation_false_rejections == 0 &&
                         rules->accepted_rules < GP_RULE_MAX_ACCEPTED;
        rules->accepted_rules += rule->accepted ? 1u : 0u;
    }
    accumulate_grid_rule_stats(rules,
                               GP_TEST_SEED_START,
                               GP_TEST_SAMPLES,
                               GP_RULE_SPLIT_TEST);
    summarize_grid_gate_rules(rules);
    return 1;
}

static void print_grid_rule_search_json(const GridGateRuleSet *rules)
{
    printf("{\"schema\":\"crlp.rule_search.summary.v1\",\"task\":\"grid_path\","
           "\"grammar_version\":%u,\"rule_set_hash\":\"grid-legality-budget-v2\","
           "\"proposer_feature_set\":\"structural\","
           "\"proposed_rules\":%zu,\"verified_rules\":%zu,\"max_rules\":%u,"
           "\"accepted_rules\":%zu,\"false_rejections\":%zu,"
           "\"train_rejects\":%zu,\"validation_rejects\":%zu,"
           "\"test_rejects\":%zu,"
           "\"train_false_rejections\":%zu,"
           "\"validation_false_rejections\":%zu,"
           "\"test_false_rejections\":%zu,"
           "\"proposer_loss\":%.8f}\n",
           (unsigned int)GP_RULE_SEARCH_GRAMMAR_VERSION,
           rules->count,
           rules->count,
           (unsigned int)GP_RULE_MAX_ACCEPTED,
           rules->accepted_rules,
           rules->test_false_rejections,
           rules->train_rejects,
           rules->validation_rejects,
           rules->test_rejects,
           rules->train_false_rejections,
           rules->validation_false_rejections,
           rules->test_false_rejections,
           rules->proposer_loss);
}

static void print_grid_rule_search_report(const GridGateRuleSet *rules)
{
    printf("[grid_path legality rule search]\n"
           "  grammar_version=%u proposed=%zu max_rules=%u accepted=%zu "
           "test_false_rejections=%zu\n"
           "  proposer_feature_set=structural final_mse=%.6f\n"
           "  accepted rules:",
           (unsigned int)GP_RULE_SEARCH_GRAMMAR_VERSION,
           rules->count,
           (unsigned int)GP_RULE_MAX_ACCEPTED,
           rules->accepted_rules,
           rules->test_false_rejections,
           rules->proposer_loss);
    int first = 1;
    for (size_t i = 0; i < rules->count; ++i) {
        const GridGateRule *rule = &rules->items[i];
        if (!rule->accepted) {
            continue;
        }
        printf("%s%s(train=%zu val=%zu test=%zu score=%.4f)",
               first ? " " : ", ",
               grid_rule_name(rule->kind),
               rule->train_rejects,
               rule->validation_rejects,
               rule->test_rejects,
               (double)rule->model_score);
        first = 0;
    }
    fputc('\n', stdout);
}

static NeuralNet *create_grid_model(unsigned int seed)
{
    static const size_t layers[] = {GP_FEATURE_COUNT, 48, 24, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int save_grid_checkpoint(NeuralNet *net,
                                const char *path,
                                const char *parent_hash,
                                const char *notes,
                                char out_hash[65])
{
    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = GP_FEATURE_ENCODER_VERSION;
    snprintf(meta.parent_model_hash,
             sizeof(meta.parent_model_hash),
             "%s",
             parent_hash == NULL ? "unknown" : parent_hash);
    snprintf(meta.training_trace_root, sizeof(meta.training_trace_root), "grid_path");
    snprintf(meta.notes,
             sizeof(meta.notes),
             "%s",
             notes == NULL ? "grid_path action scorer" : notes);
    snprintf(meta.feature_set, sizeof(meta.feature_set), "%s", "grid-observation-v3");
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

static GridMetrics evaluate_grid_model_with_rules(NeuralNet *net,
                                                  const GridGateRuleSet *rules,
                                                  uint64_t seed_start,
                                                  size_t samples,
                                                  nn_real *candidate_inputs,
                                                  uint32_t *action_ids,
                                                  nn_real *scores,
                                                  nn_real *probabilities)
{
    GridMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    size_t ranked[GP_ACTION_COUNT];

    for (size_t i = 0; i < samples; ++i) {
        GridState state;
        size_t target_index = GP_ACTION_COUNT;
        if (!generate_grid_state(seed_start + (uint64_t)i, &state)) {
            continue;
        }
        size_t candidate_count = rules == NULL
                                     ? fill_grid_candidates(&state,
                                                            candidate_inputs,
                                                            action_ids,
                                                            &target_index)
                                     : fill_grid_candidates_gated(&state,
                                                                  rules,
                                                                  candidate_inputs,
                                                                  action_ids,
                                                                  &target_index);
        if (candidate_count == 0 || target_index >= candidate_count ||
            !nn_score_action_rows(net,
                                  candidate_inputs,
                                  candidate_count,
                                  scores,
                                  probabilities) ||
            !nn_rank_action_scores(scores, candidate_count, ranked)) {
            continue;
        }

        size_t target_rank = candidate_count;
        for (size_t rank = 0; rank < candidate_count; ++rank) {
            if (ranked[rank] == target_index) {
                target_rank = rank;
                break;
            }
        }
        size_t selected = ranked[0];
        double confidence = (double)probabilities[selected];
        size_t bin = confidence >= 1.0 ? GP_CALIBRATION_BINS - 1
                                       : (size_t)(confidence * GP_CALIBRATION_BINS);
        if (bin >= GP_CALIBRATION_BINS) {
            bin = GP_CALIBRATION_BINS - 1;
        }

        ++metrics.samples;
        metrics.candidate_count += (double)candidate_count;
        metrics.rejected_candidates += (double)(GP_ACTION_COUNT - candidate_count);
        metrics.correct += selected == target_index ? 1u : 0u;
        metrics.nll += -log(safe_probability((double)probabilities[target_index]));
        metrics.utility += selected == target_index ? 1.0 : -0.25;
        metrics.proposal_steps += (double)target_rank + 1.0;
        metrics.baseline_steps += (double)target_index + 1.0;
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += selected == target_index ? 1.0 : 0.0;

        for (size_t rank = 0; rank <= target_rank; ++rank) {
            size_t action = (size_t)action_ids[ranked[rank]];
            metrics.verifier_nodes +=
                (double)bfs_nodes_from_cell(&state,
                                            state.sx + ACTION_DX[action],
                                            state.sy + ACTION_DY[action]);
        }
        for (size_t row = 0; row <= target_index; ++row) {
            size_t action = (size_t)action_ids[row];
            metrics.baseline_nodes +=
                (double)bfs_nodes_from_cell(&state,
                                            state.sx + ACTION_DX[action],
                                            state.sy + ACTION_DY[action]);
        }
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= sample_count;
    metrics.candidate_count /= sample_count;
    metrics.rejected_candidates /= sample_count;
    metrics.utility /= sample_count;
    metrics.proposal_steps /= sample_count;
    metrics.baseline_steps /= sample_count;
    metrics.verifier_nodes /= sample_count;
    metrics.baseline_nodes /= sample_count;
    for (size_t bin = 0; bin < GP_CALIBRATION_BINS; ++bin) {
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

static GridMetrics evaluate_grid_model(NeuralNet *net,
                                       uint64_t seed_start,
                                       size_t samples,
                                       nn_real *candidate_inputs,
                                       uint32_t *action_ids,
                                       nn_real *scores,
                                       nn_real *probabilities)
{
    return evaluate_grid_model_with_rules(net,
                                          NULL,
                                          seed_start,
                                          samples,
                                          candidate_inputs,
                                          action_ids,
                                          scores,
                                          probabilities);
}

static int train_grid_model(NeuralNet *net,
                            uint64_t seed_start,
                            size_t samples,
                            size_t epochs,
                            nn_real *candidate_inputs,
                            uint32_t *action_ids)
{
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        size_t updates = 0;
        for (size_t i = 0; i < samples; ++i) {
            GridState state;
            size_t target_index = GP_ACTION_COUNT;
            if (!generate_grid_state(seed_start + (uint64_t)i, &state)) {
                continue;
            }
            size_t candidate_count =
                fill_grid_candidates(&state, candidate_inputs, action_ids, &target_index);
            if (candidate_count == 0 || target_index >= candidate_count) {
                continue;
            }
            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       candidate_count,
                                                       target_index,
                                                       (nn_real)0.01);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
            ++updates;
        }
        printf("grid_path epoch %zu loss=%.4f samples=%zu\n",
               epoch + 1,
               updates == 0 ? 0.0 : loss / (double)updates,
               updates);
    }
    return 1;
}

static void print_grid_reliability_bins_json(const GridMetrics *metrics)
{
    fputs(",\"reliability_bins\":[", stdout);
    int first = 1;
    for (size_t bin = 0; bin < GP_CALIBRATION_BINS; ++bin) {
        if (metrics->bin_count[bin] == 0) {
            continue;
        }
        printf("%s{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
               "\"confidence\":%.8f,\"accuracy\":%.8f}",
               first ? "" : ",",
               (double)bin / (double)GP_CALIBRATION_BINS,
               (double)(bin + 1) / (double)GP_CALIBRATION_BINS,
               metrics->bin_count[bin],
               metrics->bin_confidence[bin],
               metrics->bin_accuracy[bin]);
        first = 0;
    }
    fputc(']', stdout);
}

static void print_grid_metrics(const char *label, const char *hash, GridMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    double saved = metrics.baseline_steps - metrics.proposal_steps;
    double saved_pct = metrics.baseline_steps <= 0.0 ? 0.0 : 100.0 * saved / metrics.baseline_steps;
    double node_saved = metrics.baseline_nodes - metrics.verifier_nodes;
    double node_saved_pct =
        metrics.baseline_nodes <= 0.0 ? 0.0 : 100.0 * node_saved / metrics.baseline_nodes;
    printf("[%s] hash=%s accuracy=%.2f%% nll=%.4f ece=%.4f utility=%.4f "
           "candidates=%.2f rejected=%.2f "
           "proposal_steps=%.3f baseline_steps=%.3f saved=%.1f%% "
           "verifier_nodes=%.2f baseline_nodes=%.2f node_saved=%.1f%% samples=%zu\n",
           label,
           hash == NULL ? "unknown" : hash,
           100.0 * (double)metrics.correct / samples,
           metrics.nll,
           metrics.ece,
           metrics.utility,
           metrics.candidate_count,
           metrics.rejected_candidates,
           metrics.proposal_steps,
           metrics.baseline_steps,
           saved_pct,
           metrics.verifier_nodes,
           metrics.baseline_nodes,
           node_saved_pct,
           metrics.samples);
}

static uint64_t grid_state_hash(uint64_t seed)
{
    uint64_t hash = 1469598103934665603ull;
    hash ^= seed;
    hash *= 1099511628211ull;
    return hash;
}

static int emit_grid_trace(NeuralNet *net,
                           const char *path,
                           const char *model_hash,
                           uint64_t seed_start,
                           size_t samples,
                           nn_real *candidate_inputs,
                           uint32_t *action_ids,
                           nn_real *scores,
                           nn_real *probabilities)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    size_t ranked[GP_ACTION_COUNT];
    int ok = 1;
    for (size_t i = 0; ok && i < samples; ++i) {
        uint64_t seed = seed_start + (uint64_t)i;
        GridState state;
        size_t target_index = GP_ACTION_COUNT;
        if (!generate_grid_state(seed, &state)) {
            ok = 0;
            break;
        }
        size_t candidate_count =
            fill_grid_candidates(&state, candidate_inputs, action_ids, &target_index);
        ok = candidate_count != 0 && target_index < candidate_count &&
             nn_score_action_rows(net,
                                  candidate_inputs,
                                  candidate_count,
                                  scores,
                                  probabilities) &&
             nn_rank_action_scores(scores, candidate_count, ranked);
        if (!ok) {
            break;
        }

        size_t target_rank = candidate_count;
        double verifier_nodes = 0.0;
        double baseline_nodes = 0.0;
        for (size_t rank = 0; rank < candidate_count; ++rank) {
            if (ranked[rank] == target_index) {
                target_rank = rank;
                break;
            }
        }
        for (size_t rank = 0; rank <= target_rank; ++rank) {
            size_t action = (size_t)action_ids[ranked[rank]];
            verifier_nodes +=
                (double)bfs_nodes_from_cell(&state,
                                            state.sx + ACTION_DX[action],
                                            state.sy + ACTION_DY[action]);
        }
        for (size_t row = 0; row <= target_index; ++row) {
            size_t action = (size_t)action_ids[row];
            baseline_nodes +=
                (double)bfs_nodes_from_cell(&state,
                                            state.sx + ACTION_DX[action],
                                            state.sy + ACTION_DY[action]);
        }

        char cost_json[192];
        char extra_json[256];
        snprintf(cost_json,
                 sizeof(cost_json),
                 "{\"proposal_steps\":%zu,\"baseline_steps\":%zu,"
                 "\"verifier_nodes\":%.0f,\"baseline_nodes\":%.0f}",
                 target_rank + 1,
                 target_index + 1,
                 verifier_nodes,
                 baseline_nodes);
        snprintf(extra_json,
                 sizeof(extra_json),
                 "\"start\":[%d,%d],\"goal\":[%d,%d],"
                 "\"proposal_steps\":%zu,\"baseline_steps\":%zu,"
                 "\"verifier_nodes\":%.0f,\"baseline_nodes\":%.0f",
                 state.sx,
                 state.sy,
                 state.gx,
                 state.gy,
                 target_rank + 1,
                 target_index + 1,
                 verifier_nodes,
                 baseline_nodes);

        NNTraceRecord record;
        memset(&record, 0, sizeof(record));
        record.task_name = "grid_path";
        record.model_hash_hex = model_hash == NULL ? "unknown" : model_hash;
        record.state_hash = grid_state_hash(seed);
        record.feature_encoder_version = GP_FEATURE_ENCODER_VERSION;
        record.candidate_count = candidate_count;
        record.action_ids = action_ids;
        record.scores = scores;
        record.probabilities = probabilities;
        record.selected_index = ranked[0];
        record.target_index = target_index;
        record.loss = (nn_real)(-log(safe_probability((double)probabilities[target_index])));
        record.verifier_result = ranked[0] == target_index ? "correct" : "corrected";
        record.verifier_cost_json = cost_json;
        record.seed = seed;
        record.extra_json = extra_json;
        ok = nn_trace_write_jsonl(fp, &record);
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (ok) {
        printf("wrote grid_path trace: %s rows=%zu\n", path, samples);
    }
    return ok;
}

static int grid_trace_append(GridTrace *trace, uint64_t seed, uint32_t target_action_id)
{
    if (trace->count == trace->capacity) {
        size_t next_capacity = trace->capacity == 0 ? 1024 : trace->capacity * 2;
        uint64_t *next_seeds = calloc(next_capacity, sizeof(*next_seeds));
        uint32_t *next_targets = calloc(next_capacity, sizeof(*next_targets));
        if (next_seeds == NULL || next_targets == NULL) {
            free(next_seeds);
            free(next_targets);
            return 0;
        }
        if (trace->count != 0) {
            memcpy(next_seeds, trace->seeds, trace->count * sizeof(*next_seeds));
            memcpy(next_targets,
                   trace->target_action_ids,
                   trace->count * sizeof(*next_targets));
        }
        free(trace->seeds);
        free(trace->target_action_ids);
        trace->seeds = next_seeds;
        trace->target_action_ids = next_targets;
        trace->capacity = next_capacity;
    }

    trace->seeds[trace->count] = seed;
    trace->target_action_ids[trace->count] = target_action_id;
    ++trace->count;
    return 1;
}

static void grid_trace_free(GridTrace *trace)
{
    free(trace->seeds);
    free(trace->target_action_ids);
    memset(trace, 0, sizeof(*trace));
}

static int load_grid_trace(const char *path, GridTrace *trace)
{
    memset(trace, 0, sizeof(*trace));
    strcpy(trace->parent_hash, "unknown");

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    char line[8192];
    uint32_t action_ids[GP_ACTION_COUNT];
    nn_real scores[GP_ACTION_COUNT];
    nn_real probabilities[GP_ACTION_COUNT];
    int ok = 1;
    while (ok && fgets(line, sizeof(line), fp) != NULL) {
        NNParsedTraceRecord record;
        if (!nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  GP_ACTION_COUNT)) {
            ok = 0;
            break;
        }
        if (strcmp(record.task_name, "grid_path") != 0) {
            continue;
        }

        uint8_t seen_actions[GP_ACTION_COUNT] = {0};
        for (size_t row = 0; row < record.candidate_count; ++row) {
            size_t action = GP_ACTION_COUNT;
            if (!grid_action_from_action_id(action_ids[row], &action) ||
                seen_actions[action]) {
                ok = 0;
                break;
            }
            seen_actions[action] = 1;
        }
        if (!ok || record.candidate_count != GP_ACTION_COUNT ||
            record.feature_encoder_version != GP_FEATURE_ENCODER_VERSION ||
            record.target_index >= record.candidate_count) {
            ok = 0;
            break;
        }
        if (strcmp(trace->parent_hash, "unknown") == 0 &&
            strcmp(record.model_hash_hex, "unknown") != 0) {
            strcpy(trace->parent_hash, record.model_hash_hex);
        }
        ok = grid_trace_append(trace, record.seed, action_ids[record.target_index]);
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || trace->count == 0) {
        grid_trace_free(trace);
        return 0;
    }
    return 1;
}

static int replay_grid_trace(const char *trace_path,
                             const char *out_path,
                             size_t epochs)
{
    GridTrace trace;
    if (!load_grid_trace(trace_path, &trace)) {
        fprintf(stderr, "failed to load trace: %s\n", trace_path);
        return 0;
    }

    NeuralNet *net = create_grid_model(1717);
    nn_real *candidate_inputs = calloc((size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    uint32_t *action_ids = calloc(GP_ACTION_COUNT, sizeof(*action_ids));
    if (net == NULL || candidate_inputs == NULL || action_ids == NULL) {
        nn_free(net);
        free(candidate_inputs);
        free(action_ids);
        grid_trace_free(&trace);
        return 0;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        size_t updates = 0;
        for (size_t i = 0; i < trace.count; ++i) {
            GridState state;
            size_t ignored_target = GP_ACTION_COUNT;
            if (!generate_grid_state(trace.seeds[i], &state)) {
                continue;
            }
            size_t candidate_count =
                fill_grid_candidates(&state, candidate_inputs, action_ids, &ignored_target);
            size_t target_index = grid_row_for_action_id(action_ids,
                                                         candidate_count,
                                                         trace.target_action_ids[i]);
            if (candidate_count == 0 || target_index >= candidate_count) {
                nn_free(net);
                free(candidate_inputs);
                free(action_ids);
                grid_trace_free(&trace);
                return 0;
            }
            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       candidate_count,
                                                       target_index,
                                                       (nn_real)0.01);
            if (isnan((double)step_loss)) {
                nn_free(net);
                free(candidate_inputs);
                free(action_ids);
                grid_trace_free(&trace);
                return 0;
            }
            loss += (double)step_loss;
            ++updates;
        }
        printf("grid_path replay epoch %zu loss=%.4f rows=%zu\n",
               epoch + 1,
               updates == 0 ? 0.0 : loss / (double)updates,
               updates);
    }

    char hash[65];
    int ok = save_grid_checkpoint(net,
                                  out_path,
                                  trace.parent_hash,
                                  "grid_path successor trained from trace replay",
                                  hash);
    if (ok) {
        printf("saved grid_path replay checkpoint: %s hash=%s parent=%s\n",
               out_path,
               hash,
               trace.parent_hash);
    }

    nn_free(net);
    free(candidate_inputs);
    free(action_ids);
    grid_trace_free(&trace);
    return ok;
}

static int compare_grid_models(int argc, char **argv)
{
    GridGateRuleSet gate_rules;
    if (!discover_grid_gate_rules(&gate_rules)) {
        fputs("failed to discover grid_path gate rules\n", stderr);
        return 0;
    }
    print_grid_rule_search_json(&gate_rules);
    print_grid_rule_search_report(&gate_rules);

    nn_real *candidate_inputs = calloc((size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    uint32_t *action_ids = calloc(GP_ACTION_COUNT, sizeof(*action_ids));
    nn_real *scores = calloc(GP_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(GP_ACTION_COUNT, sizeof(*probabilities));
    if (candidate_inputs == NULL || action_ids == NULL || scores == NULL ||
        probabilities == NULL) {
        free(candidate_inputs);
        free(action_ids);
        free(scores);
        free(probabilities);
        return 0;
    }

    int ok = 1;
    for (int i = 2; i < argc; ++i) {
        NNCheckpointMeta meta;
        NeuralNet *net = nn_load_checkpoint(argv[i], &meta);
        char hash[65] = "unknown";
        if (net == NULL) {
            fprintf(stderr, "failed to load model: %s\n", argv[i]);
            ok = 0;
            continue;
        }
        if (nn_input_size(net) != GP_FEATURE_COUNT ||
            meta.feature_encoder_version != GP_FEATURE_ENCODER_VERSION) {
            fprintf(stderr,
                    "unsupported grid_path checkpoint feature encoder: %s\n",
                    argv[i]);
            nn_free(net);
            ok = 0;
            continue;
        }
        nn_hash_checkpoint_hex(argv[i], hash);
        GridMetrics metrics = evaluate_grid_model(net,
                                                  GP_TEST_SEED_START,
                                                  GP_TEST_SAMPLES,
                                                  candidate_inputs,
                                                  action_ids,
                                                  scores,
                                                  probabilities);
        double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
        printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"grid_path\","
               "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
               "\"evaluation_mode\":\"raw\",\"rule_set_hash\":\"none\","
               "\"feature_encoder_version\":%u,"
               "\"benchmark_seed\":%u,\"sample_count\":%zu,"
               "\"accuracy\":%.8f,\"nll\":%.8f,\"ece\":%.8f,"
               "\"candidate_count\":%.8f,\"rejected_candidates\":%.8f,"
               "\"decision_utility\":%.8f,\"proposal_steps\":%.8f,"
               "\"baseline_steps\":%.8f,\"verifier_nodes\":%.8f,"
               "\"baseline_nodes\":%.8f,\"search_saved\":%.8f",
               hash,
               meta.parent_model_hash[0] == '\0' ? "unknown" : meta.parent_model_hash,
               meta.feature_encoder_version,
               (unsigned int)GP_TEST_SEED_START,
               metrics.samples,
               (double)metrics.correct / samples,
               metrics.nll,
               metrics.ece,
               metrics.candidate_count,
               metrics.rejected_candidates,
               metrics.utility,
               metrics.proposal_steps,
               metrics.baseline_steps,
               metrics.verifier_nodes,
               metrics.baseline_nodes,
               metrics.baseline_steps - metrics.proposal_steps);
        print_grid_reliability_bins_json(&metrics);
        printf("}\n");
        print_grid_metrics(argv[i], hash, metrics);

        GridMetrics gated_metrics = evaluate_grid_model_with_rules(net,
                                                                   &gate_rules,
                                                                   GP_TEST_SEED_START,
                                                                   GP_TEST_SAMPLES,
                                                                   candidate_inputs,
                                                                   action_ids,
                                                                   scores,
                                                                   probabilities);
        samples = gated_metrics.samples == 0 ? 1.0 : (double)gated_metrics.samples;
        printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"grid_path\","
               "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
               "\"evaluation_mode\":\"budgeted_legality_gate_model\","
               "\"rule_set_hash\":\"grid-legality-budget-v2\","
               "\"accepted_rules\":%zu,"
               "\"feature_encoder_version\":%u,"
               "\"benchmark_seed\":%u,\"sample_count\":%zu,"
               "\"accuracy\":%.8f,\"nll\":%.8f,\"ece\":%.8f,"
               "\"candidate_count\":%.8f,\"rejected_candidates\":%.8f,"
               "\"decision_utility\":%.8f,\"proposal_steps\":%.8f,"
               "\"baseline_steps\":%.8f,\"verifier_nodes\":%.8f,"
               "\"baseline_nodes\":%.8f,\"search_saved\":%.8f",
               hash,
               meta.parent_model_hash[0] == '\0' ? "unknown" : meta.parent_model_hash,
               gate_rules.accepted_rules,
               meta.feature_encoder_version,
               (unsigned int)GP_TEST_SEED_START,
               gated_metrics.samples,
               (double)gated_metrics.correct / samples,
               gated_metrics.nll,
               gated_metrics.ece,
               gated_metrics.candidate_count,
               gated_metrics.rejected_candidates,
               gated_metrics.utility,
               gated_metrics.proposal_steps,
               gated_metrics.baseline_steps,
               gated_metrics.verifier_nodes,
               gated_metrics.baseline_nodes,
               gated_metrics.baseline_steps - gated_metrics.proposal_steps);
        print_grid_reliability_bins_json(&gated_metrics);
        printf("}\n");
        print_grid_metrics("budgeted-legality-gate", hash, gated_metrics);
        nn_free(net);
    }

    free(candidate_inputs);
    free(action_ids);
    free(scores);
    free(probabilities);
    return ok;
}

static void print_usage(FILE *stream)
{
    fputs("usage: grid_path [--trace path] [--save path] "
          "[--trace-train] [--trace-seed seed] [--trace-samples count]\n"
          "       grid_path --replay trace.jsonl --out model.nnckpt [--epochs count]\n"
          "       grid_path --rule-search\n"
          "       grid_path --compare model_a.nnckpt model_b.nnckpt [...]\n",
          stream);
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--replay") == 0) {
        const char *trace_path = argc >= 3 ? argv[2] : NULL;
        const char *out_path = NULL;
        size_t epochs = GP_REPLAY_EPOCHS;

        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                out_path = argv[++i];
            } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
                if (!parse_size_text(argv[++i], &epochs)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        }
        if (trace_path == NULL || out_path == NULL) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        return replay_grid_trace(trace_path, out_path, epochs) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--compare") == 0) {
        if (argc < 4) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        return compare_grid_models(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--rule-search") == 0) {
        if (argc != 2) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        GridGateRuleSet rules;
        if (!discover_grid_gate_rules(&rules)) {
            fputs("failed to discover grid_path gate rules\n", stderr);
            return EXIT_FAILURE;
        }
        print_grid_rule_search_json(&rules);
        print_grid_rule_search_report(&rules);
        return EXIT_SUCCESS;
    }

    const char *trace_path = NULL;
    const char *save_path = NULL;
    uint64_t trace_seed_start = GP_TEST_SEED_START;
    size_t trace_samples = GP_TEST_SAMPLES;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--trace-train") == 0) {
            trace_seed_start = GP_TRAIN_SEED_START;
            trace_samples = GP_TRAIN_SAMPLES;
        } else if (strcmp(argv[i], "--trace-seed") == 0 && i + 1 < argc) {
            if (!parse_u64_text(argv[++i], &trace_seed_start)) {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--trace-samples") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &trace_samples)) {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        } else {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
    }

    NeuralNet *net = create_grid_model(42424);
    nn_real *candidate_inputs = calloc((size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    uint32_t *action_ids = calloc(GP_ACTION_COUNT, sizeof(*action_ids));
    nn_real *scores = calloc(GP_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(GP_ACTION_COUNT, sizeof(*probabilities));
    if (net == NULL || candidate_inputs == NULL || action_ids == NULL || scores == NULL ||
        probabilities == NULL) {
        nn_free(net);
        free(candidate_inputs);
        free(action_ids);
        free(scores);
        free(probabilities);
        return EXIT_FAILURE;
    }

    int ok = train_grid_model(net,
                              GP_TRAIN_SEED_START,
                              GP_TRAIN_SAMPLES,
                              4,
                              candidate_inputs,
                              action_ids);
    char hash[65] = "unknown";
    if (ok && save_path != NULL) {
        ok = save_grid_checkpoint(net, save_path, "unknown", "grid_path trained checkpoint", hash);
        if (ok) {
            printf("saved grid_path checkpoint: %s hash=%s\n", save_path, hash);
        }
    }
    if (ok && save_path == NULL) {
        strcpy(hash, "unsaved");
    }

    if (ok) {
        GridMetrics metrics = evaluate_grid_model(net,
                                                  GP_TEST_SEED_START,
                                                  GP_TEST_SAMPLES,
                                                  candidate_inputs,
                                                  action_ids,
                                                  scores,
                                                  probabilities);
        print_grid_metrics("grid_path", hash, metrics);
    }
    if (ok && trace_path != NULL) {
        ok = emit_grid_trace(net,
                             trace_path,
                             hash,
                             trace_seed_start,
                             trace_samples,
                             candidate_inputs,
                             action_ids,
                             scores,
                             probabilities);
    }

    nn_free(net);
    free(candidate_inputs);
    free(action_ids);
    free(scores);
    free(probabilities);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
