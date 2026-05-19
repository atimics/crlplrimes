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
    GP_RULE_SEARCH_GRAMMAR_VERSION = 5,
    GP_RULE_FEATURE_COUNT = 14,
    GP_RULE_COUNT = 10,
    GP_RULE_MAX_ACCEPTED = 4,
    GP_RULE_SCORER_EPOCHS = 96,
    GP_RULE_DEFAULT_SEED = 20240511u,
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

static const NNFeatureSchema GRID_PATH_SCORER_FEATURES[] = {
    {"start_x", NN_FEATURE_STATE_OBSERVATION, 1},
    {"start_y", NN_FEATURE_STATE_OBSERVATION, 1},
    {"goal_x", NN_FEATURE_STATE_OBSERVATION, 1},
    {"goal_y", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_dx_to_goal", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_dy_to_goal", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_manhattan_to_goal", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"candidate_manhattan_delta", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"action_dx", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_dy", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_00", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_01", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_02", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_03", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_04", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_05", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_06", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_07", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_08", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_09", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_10", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_11", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_12", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_13", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_14", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_15", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_16", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_17", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_18", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_19", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_20", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_21", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_22", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_23", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_24", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_25", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_26", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_27", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_28", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_29", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_30", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_31", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_32", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_33", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_34", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_35", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_36", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_37", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_38", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_39", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_40", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_41", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_42", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_43", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_44", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_45", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_46", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_47", NN_FEATURE_STATE_OBSERVATION, 1},
    {"blocked_cell_48", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_up", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_right", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_down", NN_FEATURE_STATE_OBSERVATION, 1},
    {"action_left", NN_FEATURE_STATE_OBSERVATION, 1},
};
_Static_assert(sizeof(GRID_PATH_SCORER_FEATURES) / sizeof(GRID_PATH_SCORER_FEATURES[0]) ==
                   GP_FEATURE_COUNT,
               "grid-path feature schema must match scorer feature count");

typedef struct {
    const char *name;
    size_t enabled_structural_prior_features;
    int use_manhattan_priors;
} GridFeatureSet;

static const GridFeatureSet GRID_FEATURE_SET_WITH_PRIORS = {
    "with_manhattan_priors",
    2,
    1,
};

static const GridFeatureSet GRID_FEATURE_SET_NO_PRIORS = {
    "no_manhattan_priors",
    0,
    0,
};

static const GridFeatureSet *grid_feature_set_or_default(const GridFeatureSet *feature_set)
{
    return feature_set == NULL ? &GRID_FEATURE_SET_WITH_PRIORS : feature_set;
}

static const GridFeatureSet *grid_feature_set_from_name(const char *name)
{
    if (name == NULL || name[0] == '\0' || strcmp(name, "unknown") == 0 ||
        strcmp(name, "grid-observation-v3") == 0 ||
        strcmp(name, "with-priors") == 0 ||
        strcmp(name, "with_manhattan_priors") == 0) {
        return &GRID_FEATURE_SET_WITH_PRIORS;
    }
    if (strcmp(name, "no-priors") == 0 ||
        strcmp(name, "no_manhattan_priors") == 0) {
        return &GRID_FEATURE_SET_NO_PRIORS;
    }
    return NULL;
}

static const char *grid_feature_set_leakage_policy_name(const GridFeatureSet *feature_set)
{
    const GridFeatureSet *selected = grid_feature_set_or_default(feature_set);
    return selected->enabled_structural_prior_features == 0
               ? nn_leakage_policy_name(NN_LEAKAGE_FORBIDDEN)
               : nn_leakage_policy_name(NN_LEAKAGE_PRIORS_ALLOWED);
}

static const NNExperimentSpec GRID_PATH_EXPERIMENT_SPEC = {
    .task_name = "grid_path",
    .role = "spatial shortest-path action-ranking benchmark for certified pruning outside arithmetic",
    .state_space = "7x7 obstacle grid with start and goal coordinates",
    .state_distribution =
        "deterministic grid generator split by train, validation, and held-out seed ranges",
    .action_space = "four moves: up, right, down, left",
    .candidate_enumerator =
        "enumerate all four moves, including blocked and out-of-bounds moves, before gating",
    .observable_features =
        "start/goal coordinates, candidate displacement, obstacle map, action one-hot, and Manhattan structural priors",
    .leakage_policy = NN_LEAKAGE_PRIORS_ALLOWED,
    .neural_scorer = "shared MLP scores one grid/action row and softmax ranks surviving moves",
    .grounding_authority = NN_GROUNDING_EXACT_CHECK,
    .grounding_operator = "BFS shortest-path oracle over the generated obstacle grid",
    .exact_verifier = "BFS identifies shortest-path successors and verifier node cost",
    .verifier_feedback = "target move, correctness, utility, candidate count, BFS nodes, calibration bins",
    .candidate_grammar = "four compass moves with legality and dominance gates",
    .outcome_row_schema = "crlp.trace.v1 and crlp.rule_search.summary.v1 rows keyed by stable move ids",
    .claim_scope = "shortest-path action ranking and certified move-pruning on generated 7x7 grids",
    .known_non_claims =
        "does not claim optimal planning beyond the generated finite grid family or undisclosed obstacle models",
    .rule_grammar =
        "legality, dead-end, component-trap, clear-path admissible-bound, and shortest-distance dominance predicates over grid actions",
    .certificate_checker =
        "finite validation/exhaustive geometry checks prove a rule never rejects an optimal move",
    .gate_executor = "accepted predicates remove impossible moves before neural scoring",
    .proposer_inputs =
        "structural rule descriptors: rule kind, direction, boundary relation, cell relation, topology relation, admissible-bound relation, and exact-distance relation",
    .proposer_targets = "train/validation residual marginal utility under a certified rule budget",
    .proposer_mode = NN_PROPOSER_SELECTION,
    .baseline_orderings = "neural, random, grammar-order, structural heuristic, validation oracle",
    .generalization_axis =
        "train on generated grids and test shifted obstacle densities, start/goal distances, and grid seeds",
    .trace_schema = NN_TRACE_SCHEMA,
    .replay_protocol = "stable action-id JSONL traces train replay successors with parent checkpoint hashes",
    .rl_protocol =
        "supervised scorer, strict verifier policy gradient, and KL/entropy-regularized verifier policy gradient",
    .features = GRID_PATH_SCORER_FEATURES,
    .feature_count = sizeof(GRID_PATH_SCORER_FEATURES) / sizeof(GRID_PATH_SCORER_FEATURES[0]),
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
    GP_RULE_DEAD_END_NEXT = 5,
    GP_RULE_COMPONENT_TRAP_NEXT = 6,
    GP_RULE_CLEAR_AXIS_DOMINATED_NEXT = 7,
    GP_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT = 8,
    GP_RULE_DISTANCE_DOMINATED_NEXT = 9,
} GridGateRuleKind;

typedef struct {
    GridGateRuleKind kind;
    size_t train_rejects;
    size_t validation_rejects;
    size_t test_rejects;
    size_t train_marginal_rejects;
    size_t validation_marginal_rejects;
    size_t test_marginal_rejects;
    size_t train_false_rejections;
    size_t validation_false_rejections;
    size_t test_false_rejections;
    size_t train_marginal_false_rejections;
    size_t validation_marginal_false_rejections;
    size_t test_marginal_false_rejections;
    nn_real model_score;
    size_t selection_order;
    int accepted;
} GridGateRule;

typedef enum {
    GP_RULE_FAMILY_ALL = 0,
    GP_RULE_FAMILY_CHEAP_LOCAL = 1,
    GP_RULE_FAMILY_BOUND_ONLY = 2,
    GP_RULE_FAMILY_CHEAP_BOUND = 3,
    GP_RULE_FAMILY_EXACT_DISTANCE = 4,
} GridRuleFamily;

typedef struct {
    GridGateRule items[GP_RULE_COUNT];
    size_t count;
    size_t accepted_rules;
    size_t cheap_rules;
    size_t exact_distance_rules;
    size_t accepted_cheap_rules;
    size_t accepted_exact_distance_rules;
    size_t train_rejects;
    size_t validation_rejects;
    size_t test_rejects;
    size_t train_marginal_rejects;
    size_t validation_marginal_rejects;
    size_t test_marginal_rejects;
    size_t train_false_rejections;
    size_t validation_false_rejections;
    size_t test_false_rejections;
    size_t train_marginal_false_rejections;
    size_t validation_marginal_false_rejections;
    size_t test_marginal_false_rejections;
    double proposer_loss;
} GridGateRuleSet;

typedef enum {
    GP_RULE_PROPOSER_NEURAL = 0,
    GP_RULE_PROPOSER_RANDOM = 1,
    GP_RULE_PROPOSER_GRAMMAR = 2,
    GP_RULE_PROPOSER_STRUCTURAL = 3,
    GP_RULE_PROPOSER_VALIDATION_ORACLE = 4,
} GridRuleProposerMode;

typedef struct {
    size_t max_rules;
    GridRuleProposerMode proposer_mode;
    uint64_t proposer_seed;
    GridRuleFamily rule_family;
} GridRuleSearchConfig;

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

static size_t fill_grid_candidates_with_feature_set(const GridState *state,
                                                    nn_real *candidate_inputs,
                                                    uint32_t *action_ids,
                                                    size_t *target_index,
                                                    const GridFeatureSet *feature_set)
{
    const GridFeatureSet *selected_feature_set = grid_feature_set_or_default(feature_set);
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
        if (selected_feature_set->use_manhattan_priors) {
            row[6] = (nn_real)((double)manhattan_after / (double)(2 * (GP_GRID - 1)));
            row[7] = (nn_real)((double)(manhattan_before - manhattan_after) / 2.0);
        }
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

static size_t fill_grid_candidates(const GridState *state,
                                   nn_real *candidate_inputs,
                                   uint32_t *action_ids,
                                   size_t *target_index)
{
    return fill_grid_candidates_with_feature_set(state,
                                                 candidate_inputs,
                                                 action_ids,
                                                 target_index,
                                                 &GRID_FEATURE_SET_WITH_PRIORS);
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
    case GP_RULE_DEAD_END_NEXT:
        return "reject action into local dead end";
    case GP_RULE_COMPONENT_TRAP_NEXT:
        return "reject action into component trap";
    case GP_RULE_CLEAR_AXIS_DOMINATED_NEXT:
        return "reject action when another successor has a clear straight path to goal";
    case GP_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT:
        return "reject action when another successor has a clear Manhattan path to goal";
    case GP_RULE_DISTANCE_DOMINATED_NEXT:
        return "reject action when another successor has shorter goal distance";
    }
    return "unknown";
}

static GridRuleSearchConfig default_grid_rule_search_config(void)
{
    GridRuleSearchConfig config;
    config.max_rules = GP_RULE_MAX_ACCEPTED;
    config.proposer_mode = GP_RULE_PROPOSER_NEURAL;
    config.proposer_seed = GP_RULE_DEFAULT_SEED;
    config.rule_family = GP_RULE_FAMILY_ALL;
    return config;
}

static const char *grid_rule_family_name(GridRuleFamily family)
{
    switch (family) {
    case GP_RULE_FAMILY_ALL:
        return "all";
    case GP_RULE_FAMILY_CHEAP_LOCAL:
        return "cheap_local";
    case GP_RULE_FAMILY_BOUND_ONLY:
        return "bound_only";
    case GP_RULE_FAMILY_CHEAP_BOUND:
        return "cheap_bound";
    case GP_RULE_FAMILY_EXACT_DISTANCE:
        return "exact_distance";
    }
    return "unknown";
}

static const char *grid_rule_family_cost_model(GridRuleFamily family)
{
    switch (family) {
    case GP_RULE_FAMILY_CHEAP_LOCAL:
        return "topology_predicates_no_exact_distance";
    case GP_RULE_FAMILY_BOUND_ONLY:
        return "clear_path_admissible_bounds";
    case GP_RULE_FAMILY_CHEAP_BOUND:
        return "topology_plus_clear_path_bounds";
    case GP_RULE_FAMILY_EXACT_DISTANCE:
        return "exact_bfs_distance";
    case GP_RULE_FAMILY_ALL:
        return "mixed_topology_bounds_and_exact_bfs_distance";
    }
    return "unknown";
}

static int parse_grid_rule_family(const char *text, GridRuleFamily *family)
{
    if (text == NULL || family == NULL) {
        return 0;
    }
    if (strcmp(text, "all") == 0 || strcmp(text, "mixed") == 0) {
        *family = GP_RULE_FAMILY_ALL;
        return 1;
    }
    if (strcmp(text, "cheap") == 0 || strcmp(text, "cheap-local") == 0 ||
        strcmp(text, "cheap_local") == 0 || strcmp(text, "local") == 0) {
        *family = GP_RULE_FAMILY_CHEAP_LOCAL;
        return 1;
    }
    if (strcmp(text, "bound") == 0 || strcmp(text, "bounds") == 0 ||
        strcmp(text, "bound-only") == 0 || strcmp(text, "bound_only") == 0 ||
        strcmp(text, "clear-bound") == 0 || strcmp(text, "clear_bound") == 0) {
        *family = GP_RULE_FAMILY_BOUND_ONLY;
        return 1;
    }
    if (strcmp(text, "cheap-bound") == 0 || strcmp(text, "cheap_bound") == 0 ||
        strcmp(text, "topology-bound") == 0 || strcmp(text, "topology_bound") == 0) {
        *family = GP_RULE_FAMILY_CHEAP_BOUND;
        return 1;
    }
    if (strcmp(text, "exact") == 0 || strcmp(text, "exact-distance") == 0 ||
        strcmp(text, "exact_distance") == 0 || strcmp(text, "distance") == 0) {
        *family = GP_RULE_FAMILY_EXACT_DISTANCE;
        return 1;
    }
    return 0;
}

static size_t grid_rule_family_rule_limit(GridRuleFamily family)
{
    switch (family) {
    case GP_RULE_FAMILY_ALL:
        return GP_RULE_COUNT;
    case GP_RULE_FAMILY_CHEAP_LOCAL:
        return 7u;
    case GP_RULE_FAMILY_BOUND_ONLY:
        return 2u;
    case GP_RULE_FAMILY_CHEAP_BOUND:
        return GP_RULE_COUNT - 1u;
    case GP_RULE_FAMILY_EXACT_DISTANCE:
        return 1u;
    }
    return GP_RULE_COUNT;
}

static int grid_gate_rule_uses_exact_distance(const GridGateRule *rule)
{
    return rule != NULL && rule->kind == GP_RULE_DISTANCE_DOMINATED_NEXT;
}

static int grid_gate_rule_allowed_in_family(GridGateRuleKind kind, GridRuleFamily family)
{
    int exact_distance = kind == GP_RULE_DISTANCE_DOMINATED_NEXT;
    int bound_rule = kind == GP_RULE_CLEAR_AXIS_DOMINATED_NEXT ||
                     kind == GP_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT;
    switch (family) {
    case GP_RULE_FAMILY_ALL:
        return 1;
    case GP_RULE_FAMILY_CHEAP_LOCAL:
        return !exact_distance && !bound_rule;
    case GP_RULE_FAMILY_BOUND_ONLY:
        return bound_rule;
    case GP_RULE_FAMILY_CHEAP_BOUND:
        return !exact_distance;
    case GP_RULE_FAMILY_EXACT_DISTANCE:
        return exact_distance;
    }
    return 0;
}

static const char *grid_rule_set_hash(const GridRuleSearchConfig *config)
{
    GridRuleFamily family = config == NULL ? GP_RULE_FAMILY_ALL : config->rule_family;
    switch (family) {
    case GP_RULE_FAMILY_CHEAP_LOCAL:
        return "grid-cheap-local-budget-v5";
    case GP_RULE_FAMILY_BOUND_ONLY:
        return "grid-bound-budget-v5";
    case GP_RULE_FAMILY_CHEAP_BOUND:
        return "grid-cheap-bound-budget-v5";
    case GP_RULE_FAMILY_EXACT_DISTANCE:
        return "grid-exact-distance-budget-v5";
    case GP_RULE_FAMILY_ALL:
        return "grid-distance-budget-v5";
    }
    return "grid-unknown-budget-v5";
}

static const char *grid_rule_proposer_mode_name(GridRuleProposerMode mode)
{
    switch (mode) {
    case GP_RULE_PROPOSER_NEURAL:
        return "neural_marginal_greedy";
    case GP_RULE_PROPOSER_RANDOM:
        return "random";
    case GP_RULE_PROPOSER_GRAMMAR:
        return "grammar_order";
    case GP_RULE_PROPOSER_STRUCTURAL:
        return "structural_heuristic";
    case GP_RULE_PROPOSER_VALIDATION_ORACLE:
        return "validation_oracle";
    }
    return "unknown";
}

static const char *grid_rule_proposer_feature_set_name(GridRuleProposerMode mode)
{
    switch (mode) {
    case GP_RULE_PROPOSER_NEURAL:
        return "structural";
    case GP_RULE_PROPOSER_RANDOM:
    case GP_RULE_PROPOSER_GRAMMAR:
        return "none";
    case GP_RULE_PROPOSER_STRUCTURAL:
        return "structural_heuristic";
    case GP_RULE_PROPOSER_VALIDATION_ORACLE:
        return "validation_counts";
    }
    return "unknown";
}

static const char *grid_rule_proposer_target_name(GridRuleProposerMode mode)
{
    switch (mode) {
    case GP_RULE_PROPOSER_NEURAL:
        return "train_validation_marginal_utility";
    case GP_RULE_PROPOSER_VALIDATION_ORACLE:
        return "validation_marginal_utility";
    case GP_RULE_PROPOSER_RANDOM:
    case GP_RULE_PROPOSER_GRAMMAR:
    case GP_RULE_PROPOSER_STRUCTURAL:
        return "none";
    }
    return "unknown";
}

static int parse_grid_rule_proposer_mode(const char *text, GridRuleProposerMode *mode)
{
    if (strcmp(text, "neural") == 0 || strcmp(text, "neural-structural") == 0 ||
        strcmp(text, "neural-marginal") == 0 ||
        strcmp(text, "neural-marginal-greedy") == 0) {
        *mode = GP_RULE_PROPOSER_NEURAL;
        return 1;
    }
    if (strcmp(text, "random") == 0) {
        *mode = GP_RULE_PROPOSER_RANDOM;
        return 1;
    }
    if (strcmp(text, "grammar") == 0 || strcmp(text, "grammar-order") == 0) {
        *mode = GP_RULE_PROPOSER_GRAMMAR;
        return 1;
    }
    if (strcmp(text, "structural") == 0 || strcmp(text, "structural-heuristic") == 0) {
        *mode = GP_RULE_PROPOSER_STRUCTURAL;
        return 1;
    }
    if (strcmp(text, "validation-oracle") == 0 || strcmp(text, "oracle") == 0) {
        *mode = GP_RULE_PROPOSER_VALIDATION_ORACLE;
        return 1;
    }
    return 0;
}

static void init_grid_gate_rules(GridGateRuleSet *rules, GridRuleFamily family)
{
    memset(rules, 0, sizeof(*rules));
    for (size_t i = 0; i < (size_t)GP_RULE_COUNT; ++i) {
        GridGateRuleKind kind = (GridGateRuleKind)i;
        if (!grid_gate_rule_allowed_in_family(kind, family)) {
            continue;
        }
        rules->items[rules->count].kind = kind;
        rules->items[rules->count].selection_order = GP_RULE_COUNT + rules->count;
        ++rules->count;
    }
}

static int grid_gate_rule_has_certificate(const GridGateRule *rule)
{
    return rule != NULL && rule->kind >= GP_RULE_UP_AT_TOP &&
           rule->kind <= GP_RULE_DISTANCE_DOMINATED_NEXT;
}

static int grid_action_next_distance(const GridState *state,
                                     const int *dist,
                                     size_t action,
                                     int *next_dist)
{
    if (state == NULL || dist == NULL || action >= GP_ACTION_COUNT ||
        next_dist == NULL) {
        return 0;
    }

    int nx = state->sx + ACTION_DX[action];
    int ny = state->sy + ACTION_DY[action];
    if (!in_bounds(nx, ny)) {
        return 0;
    }

    int next_cell = cell_index(nx, ny);
    if (state->blocked[next_cell]) {
        return 0;
    }

    *next_dist = dist[next_cell];
    return 1;
}

static int grid_best_successor_distance(const GridState *state, const int *dist)
{
    int best_dist = GP_INF;
    for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
        int next_dist = GP_INF;
        if (grid_action_next_distance(state, dist, action, &next_dist) &&
            next_dist < best_dist) {
            best_dist = next_dist;
        }
    }
    return best_dist;
}

static int grid_action_is_local_dead_end(const GridState *state, size_t action)
{
    if (state == NULL || action >= GP_ACTION_COUNT) {
        return 0;
    }

    int nx = state->sx + ACTION_DX[action];
    int ny = state->sy + ACTION_DY[action];
    if (!in_bounds(nx, ny)) {
        return 0;
    }

    int next_cell = cell_index(nx, ny);
    if (state->blocked[next_cell] || (nx == state->gx && ny == state->gy)) {
        return 0;
    }

    for (size_t exit_action = 0; exit_action < GP_ACTION_COUNT; ++exit_action) {
        int ex = nx + ACTION_DX[exit_action];
        int ey = ny + ACTION_DY[exit_action];
        if (!in_bounds(ex, ey) || state->blocked[cell_index(ex, ey)]) {
            continue;
        }
        if (ex != state->sx || ey != state->sy) {
            return 0;
        }
    }
    return 1;
}

static int grid_action_is_component_trap(const GridState *state, size_t action)
{
    if (state == NULL || action >= GP_ACTION_COUNT) {
        return 0;
    }

    int nx = state->sx + ACTION_DX[action];
    int ny = state->sy + ACTION_DY[action];
    if (!in_bounds(nx, ny)) {
        return 0;
    }

    int next_cell = cell_index(nx, ny);
    int start_cell = cell_index(state->sx, state->sy);
    int goal_cell = cell_index(state->gx, state->gy);
    if (state->blocked[next_cell] || next_cell == goal_cell) {
        return 0;
    }

    uint8_t seen[GP_CELLS] = {0};
    int queue[GP_CELLS];
    size_t head = 0;
    size_t tail = 0;

    seen[start_cell] = 1;
    seen[next_cell] = 1;
    queue[tail++] = next_cell;

    while (head < tail) {
        int cell = queue[head++];
        if (cell == goal_cell) {
            return 0;
        }
        int x = cell % GP_GRID;
        int y = cell / GP_GRID;
        for (size_t exit_action = 0; exit_action < GP_ACTION_COUNT; ++exit_action) {
            int ex = x + ACTION_DX[exit_action];
            int ey = y + ACTION_DY[exit_action];
            if (!in_bounds(ex, ey)) {
                continue;
            }
            int exit_cell = cell_index(ex, ey);
            if (state->blocked[exit_cell] || seen[exit_cell]) {
                continue;
            }
            seen[exit_cell] = 1;
            queue[tail++] = exit_cell;
        }
    }

    return 1;
}

static int grid_step_toward(int current, int target)
{
    if (current < target) {
        return 1;
    }
    if (current > target) {
        return -1;
    }
    return 0;
}

static int grid_cell_is_open(const GridState *state, int x, int y)
{
    return state != NULL && in_bounds(x, y) && !state->blocked[cell_index(x, y)];
}

static int grid_clear_manhattan_path_order(const GridState *state,
                                           int x,
                                           int y,
                                           int horizontal_first)
{
    int cx = x;
    int cy = y;
    int step_x = grid_step_toward(cx, state->gx);
    int step_y = grid_step_toward(cy, state->gy);

    if (!grid_cell_is_open(state, cx, cy)) {
        return 0;
    }

    if (horizontal_first) {
        while (cx != state->gx) {
            cx += step_x;
            if (!grid_cell_is_open(state, cx, cy)) {
                return 0;
            }
        }
        while (cy != state->gy) {
            cy += step_y;
            if (!grid_cell_is_open(state, cx, cy)) {
                return 0;
            }
        }
    } else {
        while (cy != state->gy) {
            cy += step_y;
            if (!grid_cell_is_open(state, cx, cy)) {
                return 0;
            }
        }
        while (cx != state->gx) {
            cx += step_x;
            if (!grid_cell_is_open(state, cx, cy)) {
                return 0;
            }
        }
    }

    return 1;
}

static int grid_action_clear_path_upper_bound(const GridState *state,
                                              size_t action,
                                              int require_axis,
                                              int *upper_bound)
{
    if (state == NULL || action >= GP_ACTION_COUNT || upper_bound == NULL) {
        return 0;
    }

    int nx = state->sx + ACTION_DX[action];
    int ny = state->sy + ACTION_DY[action];
    if (!grid_cell_is_open(state, nx, ny)) {
        return 0;
    }

    int manhattan = abs(state->gx - nx) + abs(state->gy - ny);
    if (require_axis && nx != state->gx && ny != state->gy) {
        return 0;
    }
    if (!grid_clear_manhattan_path_order(state, nx, ny, 1) &&
        !grid_clear_manhattan_path_order(state, nx, ny, 0)) {
        return 0;
    }

    *upper_bound = manhattan;
    return 1;
}

static int grid_action_is_clear_path_dominated(const GridState *state,
                                               size_t action,
                                               int require_axis_bound)
{
    if (state == NULL || action >= GP_ACTION_COUNT) {
        return 0;
    }

    int nx = state->sx + ACTION_DX[action];
    int ny = state->sy + ACTION_DY[action];
    if (!grid_cell_is_open(state, nx, ny)) {
        return 0;
    }

    int action_lower_bound = abs(state->gx - nx) + abs(state->gy - ny);
    if (action_lower_bound == 0) {
        return 0;
    }

    for (size_t other = 0; other < GP_ACTION_COUNT; ++other) {
        int other_upper_bound = GP_INF;
        if (other == action ||
            !grid_action_clear_path_upper_bound(state,
                                                other,
                                                require_axis_bound,
                                                &other_upper_bound)) {
            continue;
        }
        if (action_lower_bound > other_upper_bound) {
            return 1;
        }
    }
    return 0;
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
    case GP_RULE_DEAD_END_NEXT:
        return grid_action_is_local_dead_end(state, action);
    case GP_RULE_COMPONENT_TRAP_NEXT:
        return grid_action_is_component_trap(state, action);
    case GP_RULE_CLEAR_AXIS_DOMINATED_NEXT:
        return grid_action_is_clear_path_dominated(state, action, 1);
    case GP_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT:
        return grid_action_is_clear_path_dominated(state, action, 0);
    case GP_RULE_DISTANCE_DOMINATED_NEXT: {
        int dist[GP_CELLS];
        int next_dist = GP_INF;
        bfs_distances_from_goal(state, dist);
        if (!grid_action_next_distance(state, dist, action, &next_dist)) {
            return 0;
        }
        if (next_dist >= GP_INF || grid_action_is_local_dead_end(state, action)) {
            return 0;
        }
        int best_dist = grid_best_successor_distance(state, dist);
        return best_dist < GP_INF && next_dist > best_dist;
    }
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

static size_t fill_grid_candidates_gated_with_feature_set(const GridState *state,
                                                          const GridGateRuleSet *rules,
                                                          nn_real *candidate_inputs,
                                                          uint32_t *action_ids,
                                                          size_t *target_index,
                                                          const GridFeatureSet *feature_set)
{
    nn_real raw_inputs[(size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT];
    uint32_t raw_action_ids[GP_ACTION_COUNT];
    size_t raw_target = GP_ACTION_COUNT;
    fill_grid_candidates_with_feature_set(state,
                                          raw_inputs,
                                          raw_action_ids,
                                          &raw_target,
                                          feature_set);

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

static void increment_grid_rule_marginal_stats(GridGateRuleSet *rules,
                                               size_t rule_index,
                                               int false_rejection,
                                               GridRuleSplit split)
{
    GridGateRule *rule = &rules->items[rule_index];
    if (split == GP_RULE_SPLIT_TRAIN) {
        if (false_rejection) {
            ++rule->train_marginal_false_rejections;
        } else {
            ++rule->train_marginal_rejects;
        }
    } else if (split == GP_RULE_SPLIT_VALIDATION) {
        if (false_rejection) {
            ++rule->validation_marginal_false_rejections;
        } else {
            ++rule->validation_marginal_rejects;
        }
    } else {
        if (false_rejection) {
            ++rule->test_marginal_false_rejections;
        } else {
            ++rule->test_marginal_rejects;
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

static void accumulate_grid_rule_marginal_stats(GridGateRuleSet *rules,
                                                uint64_t seed_start,
                                                size_t samples,
                                                GridRuleSplit split)
{
    for (size_t i = 0; i < samples; ++i) {
        GridState state;
        size_t target_index = GP_ACTION_COUNT;
        nn_real candidate_inputs[(size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT];
        uint32_t action_ids[GP_ACTION_COUNT];
        uint8_t already_rejected[GP_ACTION_COUNT] = {0};

        if (!generate_grid_state(seed_start + (uint64_t)i, &state)) {
            continue;
        }
        fill_grid_candidates(&state, candidate_inputs, action_ids, &target_index);

        for (size_t rule_index = 0; rule_index < rules->count; ++rule_index) {
            GridGateRule *rule = &rules->items[rule_index];
            if (!rule->accepted) {
                continue;
            }
            for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
                if (already_rejected[action] ||
                    !grid_gate_rule_rejects(rule, &state, action)) {
                    continue;
                }
                already_rejected[action] = 1u;
                increment_grid_rule_marginal_stats(rules,
                                                   rule_index,
                                                   action == target_index,
                                                   split);
            }
        }
    }
}

static int fill_grid_rule_rejection_matrix(const GridGateRuleSet *rules,
                                           uint64_t seed_start,
                                           size_t samples,
                                           uint8_t *rejects,
                                           uint8_t *target_mask)
{
    if (rules == NULL || rejects == NULL || target_mask == NULL) {
        return 0;
    }

    size_t events = samples * (size_t)GP_ACTION_COUNT;
    memset(rejects, 0, rules->count * events * sizeof(*rejects));
    memset(target_mask, 0, events * sizeof(*target_mask));

    for (size_t sample = 0; sample < samples; ++sample) {
        GridState state;
        size_t target_index = GP_ACTION_COUNT;
        nn_real candidate_inputs[(size_t)GP_ACTION_COUNT * GP_FEATURE_COUNT];
        uint32_t action_ids[GP_ACTION_COUNT];
        if (!generate_grid_state(seed_start + (uint64_t)sample, &state)) {
            continue;
        }
        fill_grid_candidates(&state, candidate_inputs, action_ids, &target_index);

        for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
            size_t event = sample * (size_t)GP_ACTION_COUNT + action;
            target_mask[event] = action == target_index ? 1u : 0u;
        }
        for (size_t rule_index = 0; rule_index < rules->count; ++rule_index) {
            const GridGateRule *rule = &rules->items[rule_index];
            uint8_t *rule_rejects = rejects + rule_index * events;
            for (size_t action = 0; action < GP_ACTION_COUNT; ++action) {
                size_t event = sample * (size_t)GP_ACTION_COUNT + action;
                rule_rejects[event] =
                    grid_gate_rule_rejects(rule, &state, action) ? 1u : 0u;
            }
        }
    }
    return 1;
}

static size_t grid_rule_residual_gain(const uint8_t *rejects,
                                      const uint8_t *target_mask,
                                      const uint8_t *covered,
                                      size_t events,
                                      size_t *false_rejections)
{
    size_t gain = 0;
    size_t false_count = 0;
    for (size_t event = 0; event < events; ++event) {
        if (covered[event] || !rejects[event]) {
            continue;
        }
        if (target_mask[event]) {
            ++false_count;
        } else {
            ++gain;
        }
    }
    if (false_rejections != NULL) {
        *false_rejections = false_count;
    }
    return gain;
}

static void grid_rule_mark_residual_covered(const uint8_t *rejects,
                                            uint8_t *covered,
                                            size_t events)
{
    for (size_t event = 0; event < events; ++event) {
        if (rejects[event]) {
            covered[event] = 1u;
        }
    }
}

static int compute_grid_rule_marginal_targets(GridGateRuleSet *rules,
                                              nn_real *targets)
{
    if (rules == NULL || targets == NULL) {
        return 0;
    }

    size_t train_events = (size_t)GP_TRAIN_SAMPLES * (size_t)GP_ACTION_COUNT;
    size_t validation_events =
        (size_t)GP_VALIDATION_SAMPLES * (size_t)GP_ACTION_COUNT;
    uint8_t *train_rejects = calloc(rules->count * train_events, sizeof(*train_rejects));
    uint8_t *validation_rejects =
        calloc(rules->count * validation_events, sizeof(*validation_rejects));
    uint8_t *train_targets = calloc(train_events, sizeof(*train_targets));
    uint8_t *validation_targets = calloc(validation_events, sizeof(*validation_targets));
    uint8_t *covered_train = calloc(train_events, sizeof(*covered_train));
    uint8_t *covered_validation = calloc(validation_events, sizeof(*covered_validation));
    uint8_t selected[GP_RULE_COUNT] = {0};
    double raw_targets[GP_RULE_COUNT] = {0.0};
    double max_target = 1.0;

    if (train_rejects == NULL || validation_rejects == NULL || train_targets == NULL ||
        validation_targets == NULL || covered_train == NULL ||
        covered_validation == NULL) {
        free(train_rejects);
        free(validation_rejects);
        free(train_targets);
        free(validation_targets);
        free(covered_train);
        free(covered_validation);
        return 0;
    }

    int ok = fill_grid_rule_rejection_matrix(rules,
                                             GP_TRAIN_SEED_START,
                                             GP_TRAIN_SAMPLES,
                                             train_rejects,
                                             train_targets) &&
             fill_grid_rule_rejection_matrix(rules,
                                             GP_VALIDATION_SEED_START,
                                             GP_VALIDATION_SAMPLES,
                                             validation_rejects,
                                             validation_targets);
    if (!ok) {
        free(train_rejects);
        free(validation_rejects);
        free(train_targets);
        free(validation_targets);
        free(covered_train);
        free(covered_validation);
        return 0;
    }

    for (size_t slot = 0; slot < rules->count; ++slot) {
        size_t best_index = rules->count;
        size_t best_gain = 0;
        for (size_t rule_index = 0; rule_index < rules->count; ++rule_index) {
            GridGateRule *rule = &rules->items[rule_index];
            if (selected[rule_index] || !grid_gate_rule_has_certificate(rule) ||
                rule->train_rejects == 0 || rule->train_false_rejections != 0 ||
                rule->validation_false_rejections != 0) {
                continue;
            }

            size_t train_false = 0;
            size_t validation_false = 0;
            size_t train_gain =
                grid_rule_residual_gain(train_rejects + rule_index * train_events,
                                        train_targets,
                                        covered_train,
                                        train_events,
                                        &train_false);
            size_t validation_gain = grid_rule_residual_gain(
                validation_rejects + rule_index * validation_events,
                validation_targets,
                covered_validation,
                validation_events,
                &validation_false);
            size_t combined_gain = train_gain + validation_gain;
            if (train_false != 0 || validation_false != 0 || combined_gain == 0) {
                continue;
            }
            if (best_index == rules->count || combined_gain > best_gain ||
                (combined_gain == best_gain &&
                 rules->items[rule_index].model_score > rules->items[best_index].model_score) ||
                (combined_gain == best_gain &&
                 rules->items[rule_index].model_score == rules->items[best_index].model_score &&
                 rule->kind < rules->items[best_index].kind)) {
                best_index = rule_index;
                best_gain = combined_gain;
            }
        }
        if (best_index == rules->count) {
            break;
        }
        selected[best_index] = 1u;
        raw_targets[best_index] = (double)best_gain;
        if (raw_targets[best_index] > max_target) {
            max_target = raw_targets[best_index];
        }
        grid_rule_mark_residual_covered(train_rejects + best_index * train_events,
                                        covered_train,
                                        train_events);
        grid_rule_mark_residual_covered(validation_rejects + best_index * validation_events,
                                        covered_validation,
                                        validation_events);
    }

    for (size_t i = 0; i < rules->count; ++i) {
        targets[i] = (nn_real)(raw_targets[i] / max_target);
    }

    free(train_rejects);
    free(validation_rejects);
    free(train_targets);
    free(validation_targets);
    free(covered_train);
    free(covered_validation);
    return 1;
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
    int uses_local_topology = 0;
    int uses_admissible_bound = 0;
    int uses_axis_bound = 0;
    int uses_distance_order = 0;

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
    case GP_RULE_DEAD_END_NEXT:
        action = 0;
        observes_map = 1;
        uses_local_topology = 1;
        break;
    case GP_RULE_COMPONENT_TRAP_NEXT:
        action = 0;
        observes_map = 1;
        uses_local_topology = 1;
        break;
    case GP_RULE_CLEAR_AXIS_DOMINATED_NEXT:
        action = 0;
        observes_map = 1;
        uses_admissible_bound = 1;
        uses_axis_bound = 1;
        break;
    case GP_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT:
        action = 0;
        observes_map = 1;
        uses_admissible_bound = 1;
        break;
    case GP_RULE_DISTANCE_DOMINATED_NEXT:
        action = 0;
        uses_distance_order = 1;
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
    features[10] = uses_local_topology ? (nn_real)1 : (nn_real)0;
    features[11] = uses_admissible_bound ? (nn_real)1 : (nn_real)0;
    features[12] = uses_axis_bound ? (nn_real)1 : (nn_real)0;
    features[13] = uses_distance_order ? (nn_real)1 : (nn_real)0;
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

static uint64_t grid_rule_hash(const GridGateRule *rule, uint64_t seed)
{
    uint64_t hash = 1469598103934665603ull ^ seed;
    hash ^= (uint64_t)rule->kind + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    hash *= 1099511628211ull;
    return hash;
}

static double grid_rule_structural_score(const GridGateRule *rule)
{
    double score = grid_gate_rule_has_certificate(rule) ? 4.0 : 0.0;
    if (rule->train_false_rejections == 0 && rule->validation_false_rejections == 0) {
        score += 2.0;
    }
    if (rule->kind == GP_RULE_BLOCKED_NEXT) {
        score += 0.5;
    } else if (rule->kind == GP_RULE_CLEAR_AXIS_DOMINATED_NEXT) {
        score += 0.7;
    } else if (rule->kind == GP_RULE_CLEAR_MANHATTAN_DOMINATED_NEXT) {
        score += 0.8;
    } else if (rule->kind == GP_RULE_DISTANCE_DOMINATED_NEXT) {
        score += 1.0;
    }
    score += (double)rule->train_rejects * 0.001;
    return score;
}

static int score_grid_gate_rules(GridGateRuleSet *rules, const GridRuleSearchConfig *config)
{
    nn_real inputs[(size_t)GP_RULE_COUNT * GP_RULE_FEATURE_COUNT];
    nn_real targets[GP_RULE_COUNT];
    nn_real outputs[GP_RULE_COUNT];

    if (config->proposer_mode != GP_RULE_PROPOSER_NEURAL) {
        rules->proposer_loss = 0.0;
        for (size_t i = 0; i < rules->count; ++i) {
            GridGateRule *rule = &rules->items[i];
            switch (config->proposer_mode) {
            case GP_RULE_PROPOSER_RANDOM:
                rule->model_score =
                    (nn_real)((grid_rule_hash(rule, config->proposer_seed) >> 11) *
                              (1.0 / 9007199254740992.0));
                break;
            case GP_RULE_PROPOSER_GRAMMAR:
                rule->model_score = (nn_real)(rules->count - i);
                break;
            case GP_RULE_PROPOSER_STRUCTURAL:
                rule->model_score = (nn_real)grid_rule_structural_score(rule);
                break;
            case GP_RULE_PROPOSER_VALIDATION_ORACLE:
                rule->model_score = rule->validation_false_rejections == 0
                                        ? (nn_real)rule->validation_rejects
                                        : (nn_real)0;
                break;
            case GP_RULE_PROPOSER_NEURAL:
                break;
            }
        }
        return 1;
    }

    if (!compute_grid_rule_marginal_targets(rules, targets)) {
        return 0;
    }
    for (size_t i = 0; i < rules->count; ++i) {
        fill_grid_rule_features(&rules->items[i], &inputs[i * GP_RULE_FEATURE_COUNT]);
    }

    const size_t layers[] = {GP_RULE_FEATURE_COUNT, 8, 1};
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    nn_config.seed = config->proposer_seed == GP_RULE_DEFAULT_SEED
                         ? 31415u
                         : (31415u ^ (unsigned int)config->proposer_seed ^
                            (unsigned int)(config->proposer_seed >> 32));
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&nn_config);
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

static int compare_grid_gate_rules_by_selection(const void *left, const void *right)
{
    const GridGateRule *a = (const GridGateRule *)left;
    const GridGateRule *b = (const GridGateRule *)right;
    if (a->accepted != b->accepted) {
        return b->accepted - a->accepted;
    }
    if (a->accepted && b->accepted) {
        if (a->selection_order < b->selection_order) {
            return -1;
        }
        if (a->selection_order > b->selection_order) {
            return 1;
        }
    }
    return compare_grid_gate_rules(left, right);
}

static double grid_rule_selection_primary_score(GridRuleProposerMode mode,
                                                const GridGateRule *rule,
                                                size_t train_gain,
                                                size_t validation_gain)
{
    switch (mode) {
    case GP_RULE_PROPOSER_NEURAL:
        return (double)(train_gain + validation_gain);
    case GP_RULE_PROPOSER_VALIDATION_ORACLE:
        return (double)validation_gain;
    case GP_RULE_PROPOSER_RANDOM:
    case GP_RULE_PROPOSER_GRAMMAR:
    case GP_RULE_PROPOSER_STRUCTURAL:
        return (double)rule->model_score;
    }
    return (double)(train_gain + validation_gain);
}

static int select_grid_gate_rules(GridGateRuleSet *rules,
                                  const GridRuleSearchConfig *config)
{
    if (rules == NULL || config == NULL) {
        return 0;
    }

    size_t max_rules = config->max_rules < rules->count ? config->max_rules : rules->count;
    size_t train_events = (size_t)GP_TRAIN_SAMPLES * (size_t)GP_ACTION_COUNT;
    size_t validation_events =
        (size_t)GP_VALIDATION_SAMPLES * (size_t)GP_ACTION_COUNT;
    uint8_t *train_rejects = calloc(rules->count * train_events, sizeof(*train_rejects));
    uint8_t *validation_rejects =
        calloc(rules->count * validation_events, sizeof(*validation_rejects));
    uint8_t *train_targets = calloc(train_events, sizeof(*train_targets));
    uint8_t *validation_targets = calloc(validation_events, sizeof(*validation_targets));
    uint8_t *covered_train = calloc(train_events, sizeof(*covered_train));
    uint8_t *covered_validation = calloc(validation_events, sizeof(*covered_validation));

    if (train_rejects == NULL || validation_rejects == NULL || train_targets == NULL ||
        validation_targets == NULL || covered_train == NULL ||
        covered_validation == NULL) {
        free(train_rejects);
        free(validation_rejects);
        free(train_targets);
        free(validation_targets);
        free(covered_train);
        free(covered_validation);
        return 0;
    }

    int ok = fill_grid_rule_rejection_matrix(rules,
                                             GP_TRAIN_SEED_START,
                                             GP_TRAIN_SAMPLES,
                                             train_rejects,
                                             train_targets) &&
             fill_grid_rule_rejection_matrix(rules,
                                             GP_VALIDATION_SEED_START,
                                             GP_VALIDATION_SAMPLES,
                                             validation_rejects,
                                             validation_targets);
    if (!ok) {
        free(train_rejects);
        free(validation_rejects);
        free(train_targets);
        free(validation_targets);
        free(covered_train);
        free(covered_validation);
        return 0;
    }

    for (size_t i = 0; i < rules->count; ++i) {
        rules->items[i].accepted = 0;
        rules->items[i].selection_order = GP_RULE_COUNT + i;
    }
    rules->accepted_rules = 0;

    for (size_t slot = 0; slot < max_rules; ++slot) {
        size_t best_index = rules->count;
        double best_primary = -1.0;
        size_t best_marginal_gain = 0;
        double best_model_score = -1.0e300;
        for (size_t rule_index = 0; rule_index < rules->count; ++rule_index) {
            GridGateRule *rule = &rules->items[rule_index];
            if (rule->accepted || !grid_gate_rule_has_certificate(rule) ||
                rule->train_rejects == 0 || rule->train_false_rejections != 0 ||
                rule->validation_false_rejections != 0) {
                continue;
            }

            size_t train_false = 0;
            size_t validation_false = 0;
            size_t train_gain =
                grid_rule_residual_gain(train_rejects + rule_index * train_events,
                                        train_targets,
                                        covered_train,
                                        train_events,
                                        &train_false);
            size_t validation_gain = grid_rule_residual_gain(
                validation_rejects + rule_index * validation_events,
                validation_targets,
                covered_validation,
                validation_events,
                &validation_false);
            size_t marginal_gain = train_gain + validation_gain;
            if (train_false != 0 || validation_false != 0 || marginal_gain == 0) {
                continue;
            }

            double primary = grid_rule_selection_primary_score(config->proposer_mode,
                                                               rule,
                                                               train_gain,
                                                               validation_gain);
            if (primary <= 0.0) {
                continue;
            }
            if (best_index == rules->count || primary > best_primary ||
                (primary == best_primary && marginal_gain > best_marginal_gain) ||
                (primary == best_primary && marginal_gain == best_marginal_gain &&
                 (double)rule->model_score > best_model_score) ||
                (primary == best_primary && marginal_gain == best_marginal_gain &&
                 (double)rule->model_score == best_model_score &&
                 rule->kind < rules->items[best_index].kind)) {
                best_index = rule_index;
                best_primary = primary;
                best_marginal_gain = marginal_gain;
                best_model_score = (double)rule->model_score;
            }
        }
        if (best_index == rules->count) {
            break;
        }

        rules->items[best_index].accepted = 1;
        rules->items[best_index].selection_order = slot;
        ++rules->accepted_rules;
        grid_rule_mark_residual_covered(train_rejects + best_index * train_events,
                                        covered_train,
                                        train_events);
        grid_rule_mark_residual_covered(validation_rejects + best_index * validation_events,
                                        covered_validation,
                                        validation_events);
    }

    free(train_rejects);
    free(validation_rejects);
    free(train_targets);
    free(validation_targets);
    free(covered_train);
    free(covered_validation);
    return 1;
}

static void summarize_grid_gate_rules(GridGateRuleSet *rules)
{
    rules->accepted_rules = 0;
    rules->cheap_rules = 0;
    rules->exact_distance_rules = 0;
    rules->accepted_cheap_rules = 0;
    rules->accepted_exact_distance_rules = 0;
    rules->train_rejects = 0;
    rules->validation_rejects = 0;
    rules->test_rejects = 0;
    rules->train_marginal_rejects = 0;
    rules->validation_marginal_rejects = 0;
    rules->test_marginal_rejects = 0;
    rules->train_false_rejections = 0;
    rules->validation_false_rejections = 0;
    rules->test_false_rejections = 0;
    rules->train_marginal_false_rejections = 0;
    rules->validation_marginal_false_rejections = 0;
    rules->test_marginal_false_rejections = 0;

    for (size_t i = 0; i < rules->count; ++i) {
        const GridGateRule *rule = &rules->items[i];
        int exact_distance = grid_gate_rule_uses_exact_distance(rule);
        if (exact_distance) {
            ++rules->exact_distance_rules;
        } else {
            ++rules->cheap_rules;
        }
        if (!rule->accepted) {
            continue;
        }
        ++rules->accepted_rules;
        if (exact_distance) {
            ++rules->accepted_exact_distance_rules;
        } else {
            ++rules->accepted_cheap_rules;
        }
        rules->train_rejects += rule->train_rejects;
        rules->validation_rejects += rule->validation_rejects;
        rules->test_rejects += rule->test_rejects;
        rules->train_marginal_rejects += rule->train_marginal_rejects;
        rules->validation_marginal_rejects += rule->validation_marginal_rejects;
        rules->test_marginal_rejects += rule->test_marginal_rejects;
        rules->train_false_rejections += rule->train_false_rejections;
        rules->validation_false_rejections += rule->validation_false_rejections;
        rules->test_false_rejections += rule->test_false_rejections;
        rules->train_marginal_false_rejections += rule->train_marginal_false_rejections;
        rules->validation_marginal_false_rejections +=
            rule->validation_marginal_false_rejections;
        rules->test_marginal_false_rejections += rule->test_marginal_false_rejections;
    }
}

static int discover_grid_gate_rules(GridGateRuleSet *rules, const GridRuleSearchConfig *config)
{
    init_grid_gate_rules(rules, config == NULL ? GP_RULE_FAMILY_ALL : config->rule_family);
    accumulate_grid_rule_stats(rules,
                               GP_TRAIN_SEED_START,
                               GP_TRAIN_SAMPLES,
                               GP_RULE_SPLIT_TRAIN);
    accumulate_grid_rule_stats(rules,
                               GP_VALIDATION_SEED_START,
                               GP_VALIDATION_SAMPLES,
                               GP_RULE_SPLIT_VALIDATION);
    if (!score_grid_gate_rules(rules, config)) {
        return 0;
    }
    qsort(rules->items, rules->count, sizeof(*rules->items), compare_grid_gate_rules);
    if (!select_grid_gate_rules(rules, config)) {
        return 0;
    }
    qsort(rules->items,
          rules->count,
          sizeof(*rules->items),
          compare_grid_gate_rules_by_selection);
    accumulate_grid_rule_stats(rules,
                               GP_TEST_SEED_START,
                               GP_TEST_SAMPLES,
                               GP_RULE_SPLIT_TEST);
    accumulate_grid_rule_marginal_stats(rules,
                                        GP_TRAIN_SEED_START,
                                        GP_TRAIN_SAMPLES,
                                        GP_RULE_SPLIT_TRAIN);
    accumulate_grid_rule_marginal_stats(rules,
                                        GP_VALIDATION_SEED_START,
                                        GP_VALIDATION_SAMPLES,
                                        GP_RULE_SPLIT_VALIDATION);
    accumulate_grid_rule_marginal_stats(rules,
                                        GP_TEST_SEED_START,
                                        GP_TEST_SAMPLES,
                                        GP_RULE_SPLIT_TEST);
    summarize_grid_gate_rules(rules);
    return 1;
}

static void print_grid_rule_search_json(const GridGateRuleSet *rules,
                                        const GridRuleSearchConfig *config)
{
    char grammar_json[32];
    snprintf(grammar_json, sizeof(grammar_json), "%u", (unsigned int)GP_RULE_SEARCH_GRAMMAR_VERSION);
    char extra_json[1024];
    snprintf(extra_json,
             sizeof(extra_json),
             "\"rule_family\":\"%s\","
             "\"gate_cost_model\":\"%s\","
             "\"cheap_rules\":%zu,"
             "\"exact_distance_rules\":%zu,"
             "\"accepted_cheap_rules\":%zu,"
             "\"accepted_exact_distance_rules\":%zu,"
             "\"train_false_rejections\":%zu,"
             "\"validation_false_rejections\":%zu,"
             "\"test_false_rejections\":%zu,"
             "\"train_marginal_false_rejections\":%zu,"
             "\"validation_marginal_false_rejections\":%zu,"
             "\"test_marginal_false_rejections\":%zu",
             grid_rule_family_name(config->rule_family),
             grid_rule_family_cost_model(config->rule_family),
             rules->cheap_rules,
             rules->exact_distance_rules,
             rules->accepted_cheap_rules,
             rules->accepted_exact_distance_rules,
             rules->train_false_rejections,
             rules->validation_false_rejections,
             rules->test_false_rejections,
             rules->train_marginal_false_rejections,
             rules->validation_marginal_false_rejections,
             rules->test_marginal_false_rejections);
    NNRuleSearchSummary summary = {
        .task_name = "grid_path",
        .grammar_version_json = grammar_json,
        .rule_set_hash = grid_rule_set_hash(config),
        .max_rules = config->max_rules,
        .selection_strategy = grid_rule_proposer_mode_name(config->proposer_mode),
        .selection_seed = config->proposer_seed,
        .proposed_rules = rules->count,
        .verified_rules = rules->count,
        .accepted_rules = rules->accepted_rules,
        .false_rejections = rules->test_false_rejections,
        .proposer_feature_set = grid_rule_proposer_feature_set_name(config->proposer_mode),
        .proposer_target = grid_rule_proposer_target_name(config->proposer_mode),
        .train_rejects = rules->train_rejects,
        .validation_rejects = rules->validation_rejects,
        .test_rejects = rules->test_rejects,
        .train_marginal_rejects = rules->train_marginal_rejects,
        .validation_marginal_rejects = rules->validation_marginal_rejects,
        .test_marginal_rejects = rules->test_marginal_rejects,
        .proposer_loss = rules->proposer_loss,
        .extra_json = extra_json,
    };
    nn_rule_search_summary_write_json(stdout, &summary);
}

static void print_grid_rule_search_report(const GridGateRuleSet *rules,
                                          const GridRuleSearchConfig *config)
{
    printf("[grid_path certified rule search]\n"
           "  grammar_version=%u family=%s cost_model=%s strategy=%s "
           "proposed=%zu max_rules=%zu accepted=%zu test_false_rejections=%zu\n"
           "  proposer_feature_set=%s final_mse=%.6f\n"
           "  accepted rules:",
           (unsigned int)GP_RULE_SEARCH_GRAMMAR_VERSION,
           grid_rule_family_name(config->rule_family),
           grid_rule_family_cost_model(config->rule_family),
           grid_rule_proposer_mode_name(config->proposer_mode),
           rules->count,
           config->max_rules,
           rules->accepted_rules,
           rules->test_false_rejections,
           grid_rule_proposer_feature_set_name(config->proposer_mode),
           rules->proposer_loss);
    int first = 1;
    for (size_t i = 0; i < rules->count; ++i) {
        const GridGateRule *rule = &rules->items[i];
        if (!rule->accepted) {
            continue;
        }
        printf("%s%s(train=%zu val=%zu test=%zu "
               "marginal_test=%zu score=%.4f)",
               first ? " " : ", ",
               grid_rule_name(rule->kind),
               rule->train_rejects,
               rule->validation_rejects,
               rule->test_rejects,
               rule->test_marginal_rejects,
               (double)rule->model_score);
        first = 0;
    }
    fputc('\n', stdout);
}

static int run_grid_rule_search_report(const GridRuleSearchConfig *config)
{
    GridGateRuleSet rules;
    if (!discover_grid_gate_rules(&rules, config)) {
        return 0;
    }
    print_grid_rule_search_json(&rules, config);
    print_grid_rule_search_report(&rules, config);
    return 1;
}

static int run_grid_rule_baseline_sweep(const GridRuleSearchConfig *base_config,
                                        size_t random_trials)
{
    static const GridRuleProposerMode modes[] = {
        GP_RULE_PROPOSER_NEURAL,
        GP_RULE_PROPOSER_GRAMMAR,
        GP_RULE_PROPOSER_STRUCTURAL,
        GP_RULE_PROPOSER_VALIDATION_ORACLE,
    };
    if (base_config == NULL || random_trials == 0) {
        return 0;
    }

    for (size_t i = 0; i < sizeof(modes) / sizeof(modes[0]); ++i) {
        GridRuleSearchConfig config = *base_config;
        config.proposer_mode = modes[i];
        if (!run_grid_rule_search_report(&config)) {
            return 0;
        }
    }

    for (size_t trial = 0; trial < random_trials; ++trial) {
        GridRuleSearchConfig config = *base_config;
        config.proposer_mode = GP_RULE_PROPOSER_RANDOM;
        config.proposer_seed = base_config->proposer_seed + (uint64_t)trial;
        if (!run_grid_rule_search_report(&config)) {
            return 0;
        }
    }
    return 1;
}

static int run_grid_rule_budget_sweep(const GridRuleSearchConfig *base_config,
                                      const GridRuleFamily *families,
                                      size_t family_count,
                                      size_t seed_trials,
                                      size_t budget_max)
{
    static const GridRuleProposerMode deterministic_modes[] = {
        GP_RULE_PROPOSER_GRAMMAR,
        GP_RULE_PROPOSER_STRUCTURAL,
        GP_RULE_PROPOSER_VALIDATION_ORACLE,
    };
    if (base_config == NULL || families == NULL || family_count == 0 ||
        seed_trials == 0 || budget_max == 0) {
        return 0;
    }

    for (size_t family_i = 0; family_i < family_count; ++family_i) {
        size_t family_limit = grid_rule_family_rule_limit(families[family_i]);
        size_t max_budget = budget_max < family_limit ? budget_max : family_limit;
        for (size_t budget = 1; budget <= max_budget; ++budget) {
            for (size_t trial = 0; trial < seed_trials; ++trial) {
                GridRuleSearchConfig config = *base_config;
                config.rule_family = families[family_i];
                config.max_rules = budget;
                config.proposer_mode = GP_RULE_PROPOSER_NEURAL;
                config.proposer_seed = base_config->proposer_seed + (uint64_t)trial;
                if (!run_grid_rule_search_report(&config)) {
                    return 0;
                }
            }
            for (size_t mode_i = 0;
                 mode_i < sizeof(deterministic_modes) / sizeof(deterministic_modes[0]);
                 ++mode_i) {
                GridRuleSearchConfig config = *base_config;
                config.rule_family = families[family_i];
                config.max_rules = budget;
                config.proposer_mode = deterministic_modes[mode_i];
                if (!run_grid_rule_search_report(&config)) {
                    return 0;
                }
            }
            for (size_t trial = 0; trial < seed_trials; ++trial) {
                GridRuleSearchConfig config = *base_config;
                config.rule_family = families[family_i];
                config.max_rules = budget;
                config.proposer_mode = GP_RULE_PROPOSER_RANDOM;
                config.proposer_seed = base_config->proposer_seed + (uint64_t)trial;
                if (!run_grid_rule_search_report(&config)) {
                    return 0;
                }
            }
        }
    }
    return 1;
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
                                const GridFeatureSet *feature_set,
                                char out_hash[65])
{
    const GridFeatureSet *selected_feature_set = grid_feature_set_or_default(feature_set);
    NNCheckpointMeta meta;
    nn_checkpoint_meta_init(&meta, GP_FEATURE_ENCODER_VERSION);
    if (!nn_checkpoint_meta_set_parent_model_hash(&meta,
                                                  parent_hash == NULL ? "unknown" : parent_hash) ||
        !nn_checkpoint_meta_set_training_trace_root(&meta, "grid_path") ||
        !nn_checkpoint_meta_set_notes(&meta,
                                      notes == NULL ? "grid_path action scorer" : notes) ||
        !nn_checkpoint_meta_set_feature_set(&meta, selected_feature_set->name)) {
        return 0;
    }
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

static GridMetrics evaluate_grid_model_with_rules_and_features(NeuralNet *net,
                                                               const GridGateRuleSet *rules,
                                                               uint64_t seed_start,
                                                               size_t samples,
                                                               nn_real *candidate_inputs,
                                                               uint32_t *action_ids,
                                                               nn_real *scores,
                                                               nn_real *probabilities,
                                                               const GridFeatureSet *feature_set)
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
                                     ? fill_grid_candidates_with_feature_set(&state,
                                                                            candidate_inputs,
                                                                            action_ids,
                                                                            &target_index,
                                                                            feature_set)
                                     : fill_grid_candidates_gated_with_feature_set(
                                           &state,
                                           rules,
                                           candidate_inputs,
                                           action_ids,
                                           &target_index,
                                           feature_set);
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

static int train_grid_model(NeuralNet *net,
                            uint64_t seed_start,
                            size_t samples,
                            size_t epochs,
                            nn_real *candidate_inputs,
                            uint32_t *action_ids,
                            const GridFeatureSet *feature_set)
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
            size_t candidate_count = fill_grid_candidates_with_feature_set(&state,
                                                                           candidate_inputs,
                                                                           action_ids,
                                                                           &target_index,
                                                                           feature_set);
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
                                  &GRID_FEATURE_SET_WITH_PRIORS,
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

static void print_grid_feature_set_json_fields(const GridFeatureSet *feature_set)
{
    const GridFeatureSet *selected = grid_feature_set_or_default(feature_set);
    printf("\"feature_set\":\"%s\","
           "\"enabled_structural_prior_features\":%zu,"
           "\"structural_prior_features\":[",
           selected->name,
           selected->enabled_structural_prior_features);
    if (selected->use_manhattan_priors) {
        fputs("\"candidate_manhattan_to_goal\",\"candidate_manhattan_delta\"", stdout);
    }
    fputs("],", stdout);
}

static const char *grid_rule_family_name_for_hash(const char *rule_set_hash)
{
    if (rule_set_hash == NULL || strcmp(rule_set_hash, "none") == 0) {
        return "none";
    }
    if (strcmp(rule_set_hash, "grid-cheap-local-budget-v5") == 0 ||
        strcmp(rule_set_hash, "grid-cheap-local-budget-v4") == 0 ||
        strcmp(rule_set_hash, "grid-cheap-local-budget-v3") == 0) {
        return grid_rule_family_name(GP_RULE_FAMILY_CHEAP_LOCAL);
    }
    if (strcmp(rule_set_hash, "grid-bound-budget-v5") == 0) {
        return grid_rule_family_name(GP_RULE_FAMILY_BOUND_ONLY);
    }
    if (strcmp(rule_set_hash, "grid-cheap-bound-budget-v5") == 0) {
        return grid_rule_family_name(GP_RULE_FAMILY_CHEAP_BOUND);
    }
    if (strcmp(rule_set_hash, "grid-exact-distance-budget-v5") == 0 ||
        strcmp(rule_set_hash, "grid-exact-distance-budget-v4") == 0 ||
        strcmp(rule_set_hash, "grid-exact-distance-budget-v3") == 0) {
        return grid_rule_family_name(GP_RULE_FAMILY_EXACT_DISTANCE);
    }
    if (strcmp(rule_set_hash, "grid-distance-budget-v5") == 0 ||
        strcmp(rule_set_hash, "grid-distance-budget-v4") == 0 ||
        strcmp(rule_set_hash, "grid-distance-budget-v3") == 0) {
        return grid_rule_family_name(GP_RULE_FAMILY_ALL);
    }
    return "unknown";
}

static const char *grid_rule_cost_model_for_hash(const char *rule_set_hash)
{
    if (rule_set_hash == NULL || strcmp(rule_set_hash, "none") == 0) {
        return "none";
    }
    if (strcmp(rule_set_hash, "grid-cheap-local-budget-v5") == 0 ||
        strcmp(rule_set_hash, "grid-cheap-local-budget-v4") == 0) {
        return grid_rule_family_cost_model(GP_RULE_FAMILY_CHEAP_LOCAL);
    }
    if (strcmp(rule_set_hash, "grid-cheap-local-budget-v3") == 0) {
        return "local_predicates_only";
    }
    if (strcmp(rule_set_hash, "grid-bound-budget-v5") == 0) {
        return grid_rule_family_cost_model(GP_RULE_FAMILY_BOUND_ONLY);
    }
    if (strcmp(rule_set_hash, "grid-cheap-bound-budget-v5") == 0) {
        return grid_rule_family_cost_model(GP_RULE_FAMILY_CHEAP_BOUND);
    }
    if (strcmp(rule_set_hash, "grid-exact-distance-budget-v5") == 0 ||
        strcmp(rule_set_hash, "grid-exact-distance-budget-v4") == 0 ||
        strcmp(rule_set_hash, "grid-exact-distance-budget-v3") == 0) {
        return grid_rule_family_cost_model(GP_RULE_FAMILY_EXACT_DISTANCE);
    }
    if (strcmp(rule_set_hash, "grid-distance-budget-v5") == 0 ||
        strcmp(rule_set_hash, "grid-distance-budget-v4") == 0 ||
        strcmp(rule_set_hash, "grid-distance-budget-v3") == 0) {
        return grid_rule_family_cost_model(GP_RULE_FAMILY_ALL);
    }
    return "unknown";
}

static void print_grid_compare_json(const NNCheckpointMeta *meta,
                                    const char *hash,
                                    const char *evaluation_mode,
                                    const char *rule_set_hash,
                                    size_t accepted_rules,
                                    const GridFeatureSet *feature_set,
                                    const GridMetrics *metrics)
{
    const GridFeatureSet *selected_feature_set = grid_feature_set_or_default(feature_set);
    double samples = metrics->samples == 0 ? 1.0 : (double)metrics->samples;
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"grid_path\","
           "\"leakage_policy\":\"%s\","
           "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
           "\"evaluation_mode\":\"%s\",\"rule_set_hash\":\"%s\","
           "\"rule_family\":\"%s\","
           "\"gate_cost_model\":\"%s\","
           "\"accepted_rules\":%zu,",
           grid_feature_set_leakage_policy_name(selected_feature_set),
           hash == NULL ? "unknown" : hash,
           meta == NULL || meta->parent_model_hash[0] == '\0' ? "unknown"
                                                               : meta->parent_model_hash,
           evaluation_mode == NULL ? "raw" : evaluation_mode,
           rule_set_hash == NULL ? "none" : rule_set_hash,
           grid_rule_family_name_for_hash(rule_set_hash),
           grid_rule_cost_model_for_hash(rule_set_hash),
           accepted_rules);
    print_grid_feature_set_json_fields(selected_feature_set);
    printf("\"feature_encoder_version\":%u,"
           "\"benchmark_seed\":%u,\"sample_count\":%zu,"
           "\"accuracy\":%.8f,\"nll\":%.8f,\"ece\":%.8f,"
           "\"candidate_count\":%.8f,\"rejected_candidates\":%.8f,"
           "\"decision_utility\":%.8f,\"proposal_steps\":%.8f,"
           "\"baseline_steps\":%.8f,\"verifier_nodes\":%.8f,"
           "\"baseline_nodes\":%.8f,\"search_saved\":%.8f",
           meta == NULL ? (unsigned int)GP_FEATURE_ENCODER_VERSION
                        : (unsigned int)meta->feature_encoder_version,
           (unsigned int)GP_TEST_SEED_START,
           metrics->samples,
           (double)metrics->correct / samples,
           metrics->nll,
           metrics->ece,
           metrics->candidate_count,
           metrics->rejected_candidates,
           metrics->utility,
           metrics->proposal_steps,
           metrics->baseline_steps,
           metrics->verifier_nodes,
           metrics->baseline_nodes,
           metrics->baseline_steps - metrics->proposal_steps);
    print_grid_reliability_bins_json(metrics);
    printf("}\n");
}

static void print_usage(FILE *stream);

static int compare_grid_models(int argc, char **argv)
{
    GridRuleSearchConfig rule_config = default_grid_rule_search_config();
    GridRuleSearchConfig family_configs[5];
    GridGateRuleSet family_rules[5];
    size_t family_count = 0;
    int family_sweep = 0;
    const char *model_paths[64];
    size_t model_count = 0;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--rule-family") == 0 && i + 1 < argc) {
            if (!parse_grid_rule_family(argv[++i], &rule_config.rule_family)) {
                print_usage(stderr);
                return 0;
            }
            continue;
        }
        if (strcmp(argv[i], "--rule-family-sweep") == 0) {
            family_sweep = 1;
            continue;
        }
        if (model_count >= sizeof(model_paths) / sizeof(model_paths[0])) {
            fprintf(stderr, "too many grid_path checkpoints\n");
            return 0;
        }
        model_paths[model_count++] = argv[i];
    }
    if (model_count == 0) {
        print_usage(stderr);
        return 0;
    }

    if (family_sweep) {
        family_count = 5;
        for (size_t i = 0; i < family_count; ++i) {
            family_configs[i] = default_grid_rule_search_config();
        }
        family_configs[0].rule_family = GP_RULE_FAMILY_CHEAP_LOCAL;
        family_configs[1].rule_family = GP_RULE_FAMILY_BOUND_ONLY;
        family_configs[2].rule_family = GP_RULE_FAMILY_CHEAP_BOUND;
        family_configs[3].rule_family = GP_RULE_FAMILY_EXACT_DISTANCE;
        family_configs[4].rule_family = GP_RULE_FAMILY_ALL;
    } else {
        family_count = 1;
        family_configs[0] = rule_config;
    }

    for (size_t i = 0; i < family_count; ++i) {
        if (!discover_grid_gate_rules(&family_rules[i], &family_configs[i])) {
            fputs("failed to discover grid_path gate rules\n", stderr);
            return 0;
        }
        print_grid_rule_search_json(&family_rules[i], &family_configs[i]);
        print_grid_rule_search_report(&family_rules[i], &family_configs[i]);
    }

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
    for (size_t i = 0; i < model_count; ++i) {
        NNCheckpointMeta meta;
        NeuralNet *net = nn_load_checkpoint(model_paths[i], &meta);
        char hash[65] = "unknown";
        if (net == NULL) {
            fprintf(stderr, "failed to load model: %s\n", model_paths[i]);
            ok = 0;
            continue;
        }
        if (nn_input_size(net) != GP_FEATURE_COUNT ||
            meta.feature_encoder_version != GP_FEATURE_ENCODER_VERSION) {
            fprintf(stderr,
                    "unsupported grid_path checkpoint feature encoder: %s\n",
                    model_paths[i]);
            nn_free(net);
            ok = 0;
            continue;
        }
        const GridFeatureSet *checkpoint_feature_set =
            grid_feature_set_from_name(meta.feature_set);
        if (checkpoint_feature_set == NULL) {
            fprintf(stderr,
                    "unsupported grid_path checkpoint feature_set: %s\n",
                    meta.feature_set[0] == '\0' ? "unknown" : meta.feature_set);
            nn_free(net);
            ok = 0;
            continue;
        }
        nn_hash_checkpoint_hex(model_paths[i], hash);
        GridMetrics metrics = evaluate_grid_model_with_rules_and_features(
            net,
            NULL,
            GP_TEST_SEED_START,
            GP_TEST_SAMPLES,
            candidate_inputs,
            action_ids,
            scores,
            probabilities,
            checkpoint_feature_set);
        print_grid_compare_json(&meta,
                                hash,
                                "raw",
                                "none",
                                0,
                                checkpoint_feature_set,
                                &metrics);
        print_grid_metrics(model_paths[i], hash, metrics);

        if (checkpoint_feature_set->use_manhattan_priors) {
            GridMetrics no_prior_metrics = evaluate_grid_model_with_rules_and_features(
                net,
                NULL,
                GP_TEST_SEED_START,
                GP_TEST_SAMPLES,
                candidate_inputs,
                action_ids,
                scores,
                probabilities,
                &GRID_FEATURE_SET_NO_PRIORS);
            print_grid_compare_json(&meta,
                                    hash,
                                    "raw_prior_disabled_ablation",
                                    "none",
                                    0,
                                    &GRID_FEATURE_SET_NO_PRIORS,
                                    &no_prior_metrics);
            print_grid_metrics("raw-prior-disabled-ablation", hash, no_prior_metrics);
        }

        for (size_t family_i = 0; family_i < family_count; ++family_i) {
            GridMetrics gated_metrics = evaluate_grid_model_with_rules_and_features(
                net,
                &family_rules[family_i],
                GP_TEST_SEED_START,
                GP_TEST_SAMPLES,
                candidate_inputs,
                action_ids,
                scores,
                probabilities,
                checkpoint_feature_set);
            print_grid_compare_json(&meta,
                                    hash,
                                    "budgeted_legality_gate_model",
                                    grid_rule_set_hash(&family_configs[family_i]),
                                    family_rules[family_i].accepted_rules,
                                    checkpoint_feature_set,
                                    &gated_metrics);
            print_grid_metrics("budgeted-certified-gate", hash, gated_metrics);

            if (checkpoint_feature_set->use_manhattan_priors) {
                GridMetrics gated_no_prior_metrics =
                    evaluate_grid_model_with_rules_and_features(
                        net,
                        &family_rules[family_i],
                        GP_TEST_SEED_START,
                        GP_TEST_SAMPLES,
                        candidate_inputs,
                        action_ids,
                        scores,
                        probabilities,
                        &GRID_FEATURE_SET_NO_PRIORS);
                print_grid_compare_json(&meta,
                                        hash,
                                        "budgeted_legality_gate_prior_disabled_ablation",
                                        grid_rule_set_hash(&family_configs[family_i]),
                                        family_rules[family_i].accepted_rules,
                                        &GRID_FEATURE_SET_NO_PRIORS,
                                        &gated_no_prior_metrics);
                print_grid_metrics("budgeted-certified-gate-prior-disabled-ablation",
                                   hash,
                                   gated_no_prior_metrics);
            }
        }
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
          "[--feature-set with-priors|no-priors] "
          "[--trace-train] [--trace-seed seed] [--trace-samples count]\n"
          "       grid_path --replay trace.jsonl --out model.nnckpt [--epochs count]\n"
          "       grid_path --rule-search [--max-rules count] "
          "[--rule-family all|cheap|bound|cheap-bound|exact-distance] "
          "[--rule-family-sweep] "
          "[--baseline-sweep] [--budget-sweep] "
          "[--random-trials count] "
          "[--proposer neural|random|grammar|structural|validation-oracle] "
          "[--proposer-seed seed]\n"
          "       grid_path --compare "
          "[--rule-family all|cheap|bound|cheap-bound|exact-distance] "
          "[--rule-family-sweep] "
          "model_a.nnckpt model_b.nnckpt [...]\n"
          "       grid_path --schema\n",
          stream);
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--schema") == 0) {
        return nn_experiment_write_schema_json(stdout, &GRID_PATH_EXPERIMENT_SPEC)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--replay") == 0) {
        const char *trace_path = argc >= 3 ? argv[2] : NULL;
        const char *out_path = NULL;
        size_t epochs = GP_REPLAY_EPOCHS;

        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                out_path = argv[++i];
            } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
                if (!nn_parse_size_text(argv[++i], &epochs)) {
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
        GridRuleSearchConfig rule_config = default_grid_rule_search_config();
        int baseline_sweep = 0;
        int budget_sweep = 0;
        int family_sweep = 0;
        int max_rules_explicit = 0;
        size_t random_trials = 5;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--max-rules") == 0 && i + 1 < argc) {
                if (!nn_parse_size_text(argv[++i], &rule_config.max_rules) ||
                    rule_config.max_rules == 0 ||
                    rule_config.max_rules > GP_RULE_COUNT) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
                max_rules_explicit = 1;
            } else if (strcmp(argv[i], "--proposer") == 0 && i + 1 < argc) {
                if (!parse_grid_rule_proposer_mode(argv[++i], &rule_config.proposer_mode)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--proposer-seed") == 0 && i + 1 < argc) {
                if (!nn_parse_u64_text(argv[++i], &rule_config.proposer_seed)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--rule-family") == 0 && i + 1 < argc) {
                if (!parse_grid_rule_family(argv[++i], &rule_config.rule_family)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--rule-family-sweep") == 0) {
                family_sweep = 1;
            } else if (strcmp(argv[i], "--baseline-sweep") == 0) {
                baseline_sweep = 1;
            } else if (strcmp(argv[i], "--budget-sweep") == 0) {
                budget_sweep = 1;
            } else if (strcmp(argv[i], "--random-trials") == 0 && i + 1 < argc) {
                if (!nn_parse_size_text(argv[++i], &random_trials) || random_trials == 0) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        }
        GridRuleFamily families[5];
        size_t family_count = 1;
        families[0] = rule_config.rule_family;
        if (family_sweep) {
            family_count = 5;
            families[0] = GP_RULE_FAMILY_CHEAP_LOCAL;
            families[1] = GP_RULE_FAMILY_BOUND_ONLY;
            families[2] = GP_RULE_FAMILY_CHEAP_BOUND;
            families[3] = GP_RULE_FAMILY_EXACT_DISTANCE;
            families[4] = GP_RULE_FAMILY_ALL;
        }
        if (budget_sweep) {
            size_t budget_max = max_rules_explicit ? rule_config.max_rules : GP_RULE_COUNT;
            return run_grid_rule_budget_sweep(&rule_config,
                                              families,
                                              family_count,
                                              random_trials,
                                              budget_max)
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE;
        }
        if (baseline_sweep) {
            for (size_t family_i = 0; family_i < family_count; ++family_i) {
                GridRuleSearchConfig family_config = rule_config;
                family_config.rule_family = families[family_i];
                if (!run_grid_rule_baseline_sweep(&family_config, random_trials)) {
                    return EXIT_FAILURE;
                }
            }
            return EXIT_SUCCESS;
        }
        if (family_sweep) {
            for (size_t family_i = 0; family_i < family_count; ++family_i) {
                GridRuleSearchConfig family_config = rule_config;
                family_config.rule_family = families[family_i];
                if (!run_grid_rule_search_report(&family_config)) {
                    fputs("failed to discover grid_path gate rules\n", stderr);
                    return EXIT_FAILURE;
                }
            }
            return EXIT_SUCCESS;
        }
        if (!run_grid_rule_search_report(&rule_config)) {
            fputs("failed to discover grid_path gate rules\n", stderr);
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    const char *trace_path = NULL;
    const char *save_path = NULL;
    const GridFeatureSet *feature_set = &GRID_FEATURE_SET_WITH_PRIORS;
    uint64_t trace_seed_start = GP_TEST_SEED_START;
    size_t trace_samples = GP_TEST_SAMPLES;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--feature-set") == 0 && i + 1 < argc) {
            feature_set = grid_feature_set_from_name(argv[++i]);
            if (feature_set == NULL) {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--trace-train") == 0) {
            trace_seed_start = GP_TRAIN_SEED_START;
            trace_samples = GP_TRAIN_SAMPLES;
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
    if (trace_path != NULL && !feature_set->use_manhattan_priors) {
        fputs("grid_path trace/replay currently requires with-priors feature_set\n", stderr);
        return EXIT_FAILURE;
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
                              action_ids,
                              feature_set);
    char hash[65] = "unknown";
    if (ok && save_path != NULL) {
        ok = save_grid_checkpoint(net,
                                  save_path,
                                  "unknown",
                                  "grid_path trained checkpoint",
                                  feature_set,
                                  hash);
        if (ok) {
            printf("saved grid_path checkpoint: %s hash=%s\n", save_path, hash);
        }
    }
    if (ok && save_path == NULL) {
        strcpy(hash, "unsaved");
    }

    if (ok) {
        GridMetrics metrics = evaluate_grid_model_with_rules_and_features(
            net,
            NULL,
            GP_TEST_SEED_START,
            GP_TEST_SAMPLES,
            candidate_inputs,
            action_ids,
            scores,
            probabilities,
            feature_set);
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
