#include "nn.h"
#include "../src/prime_gap/internal.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    MT_TASK_PRIME_GAP = 0,
    MT_TASK_GRID_PATH = 1,
    MT_TASK_DELAYED_CLAIM = 2,
    MT_TASK_ARC_AGI = 3,
    MT_TASK_COUNT = 4,
} MTTask;

enum {
    MT_FEATURE_COUNT = 96,
    MT_MAX_CANDIDATES = ACTION_COUNT,
    MT_CALIBRATION_BINS = 10,
    MT_GRID_SIZE = 7,
    MT_GRID_CELLS = MT_GRID_SIZE * MT_GRID_SIZE,
    MT_GRID_ACTION_COUNT = 4,
    MT_GRID_INF = 1000000,
    MT_CLAIM_STATE_FEATURES = 6,
    MT_CLAIM_ACTION_COUNT = 4,
    MT_ARC_INPUT_WIDTH = 7,
    MT_ARC_INPUT_HEIGHT = 3,
    MT_ARC_INPUT_CELLS = MT_ARC_INPUT_WIDTH * MT_ARC_INPUT_HEIGHT,
    MT_ARC_OUTPUT_WIDTH = 3,
    MT_ARC_OUTPUT_HEIGHT = 3,
    MT_ARC_OUTPUT_CELLS = MT_ARC_OUTPUT_WIDTH * MT_ARC_OUTPUT_HEIGHT,
    MT_ARC_ACTION_COUNT = 10,
    MT_ARC_TRAIN_PAIR_COUNT = 3,
    MT_F_BIAS = 0,
    MT_F_TASK = 1,
    MT_F_ACTION_INDEX = 5,
    MT_F_CANDIDATE_COUNT = 6,
    MT_F_COMMON = 7,
    MT_F_STATE = 16,
    MT_F_TASK_SPECIFIC = 32,
    MT_DEFAULT_EPOCHS = 4,
    MT_DEFAULT_STEPS_PER_EPOCH = 6000,
    MT_DEFAULT_EVAL_SAMPLES = 1000,
};

static const int MT_GRID_DX[MT_GRID_ACTION_COUNT] = {0, 1, 0, -1};
static const int MT_GRID_DY[MT_GRID_ACTION_COUNT] = {-1, 0, 1, 0};
/* ARC-AGI-1 training task 0520fde7, reduced to balanced cell-color scoring. */
static const uint8_t MT_ARC_0520FDE7_TRAIN_INPUTS[MT_ARC_TRAIN_PAIR_COUNT][MT_ARC_INPUT_CELLS] = {
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
static const uint8_t MT_ARC_0520FDE7_TRAIN_OUTPUTS[MT_ARC_TRAIN_PAIR_COUNT][MT_ARC_OUTPUT_CELLS] = {
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
static const uint8_t MT_ARC_0520FDE7_TEST_INPUT[MT_ARC_INPUT_CELLS] = {
    1, 0, 1, 5, 1, 0, 1,
    0, 1, 0, 5, 1, 0, 1,
    1, 0, 1, 5, 0, 1, 0,
};
static const uint8_t MT_ARC_0520FDE7_TEST_OUTPUT[MT_ARC_OUTPUT_CELLS] = {
    2, 0, 2,
    0, 0, 0,
    0, 0, 0,
};

typedef struct {
    uint8_t blocked[MT_GRID_CELLS];
    int sx;
    int sy;
    int gx;
    int gy;
} MTGridState;

typedef struct {
    nn_real state[MT_CLAIM_STATE_FEATURES];
    size_t target;
    nn_real utility[MT_CLAIM_ACTION_COUNT];
} MTClaimState;

typedef struct {
    size_t epochs;
    size_t steps_per_epoch;
    size_t eval_samples;
    uint64_t seed;
    nn_real learning_rate;
    size_t hidden1;
    size_t hidden2;
    int run_task_baselines;
    int use_task_tags;
    uint32_t task_mask;
    uint32_t eval_task_mask;
    int training_order;
} MTConfig;

typedef struct {
    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    double prime_log_scale;
} MTData;

typedef struct {
    size_t samples;
    size_t top1;
    size_t top5;
    size_t top10;
    double nll;
    double mean_rank;
    double candidate_count;
    double utility;
    double baseline_steps;
    double ece;
    double confidence_sum;
    double correct_confidence_sum;
    double incorrect_confidence_sum;
    size_t correct_confidence_count;
    size_t incorrect_confidence_count;
    size_t bin_count[MT_CALIBRATION_BINS];
    double bin_confidence[MT_CALIBRATION_BINS];
    double bin_accuracy[MT_CALIBRATION_BINS];
} MTMetrics;

typedef enum {
    MT_TRAINING_UNIFORM_RANDOM = 0,
    MT_TRAINING_ROUND_ROBIN = 1,
    MT_TRAINING_ARC_HEAVY = 2,
    MT_TRAINING_PRIME_LIGHT = 3,
} MTTrainingOrder;

static const char *mt_task_name(MTTask task)
{
    switch (task) {
    case MT_TASK_PRIME_GAP:
        return "prime_gap";
    case MT_TASK_GRID_PATH:
        return "grid_path";
    case MT_TASK_DELAYED_CLAIM:
        return "delayed_claim";
    case MT_TASK_ARC_AGI:
        return "arc_agi";
    case MT_TASK_COUNT:
        break;
    }
    return "unknown";
}

static uint32_t mt_task_bit(MTTask task)
{
    return UINT32_C(1) << (uint32_t)task;
}

static int mt_task_from_name(const char *name, MTTask *out_task)
{
    if (strcmp(name, "prime") == 0 || strcmp(name, "prime_gap") == 0 ||
        strcmp(name, "prime-gap") == 0) {
        *out_task = MT_TASK_PRIME_GAP;
        return 1;
    }
    if (strcmp(name, "grid") == 0 || strcmp(name, "grid_path") == 0 ||
        strcmp(name, "grid-path") == 0) {
        *out_task = MT_TASK_GRID_PATH;
        return 1;
    }
    if (strcmp(name, "claim") == 0 || strcmp(name, "delayed_claim") == 0 ||
        strcmp(name, "delayed-claim") == 0) {
        *out_task = MT_TASK_DELAYED_CLAIM;
        return 1;
    }
    if (strcmp(name, "arc") == 0 || strcmp(name, "arc_agi") == 0 ||
        strcmp(name, "arc-agi") == 0 || strcmp(name, "arcagi") == 0) {
        *out_task = MT_TASK_ARC_AGI;
        return 1;
    }
    return 0;
}

static size_t mt_active_task_count(uint32_t task_mask)
{
    size_t count = 0;
    for (size_t task = 0; task < MT_TASK_COUNT; ++task) {
        if ((task_mask & mt_task_bit((MTTask)task)) != 0) {
            ++count;
        }
    }
    return count;
}

static size_t mt_fill_active_tasks(uint32_t task_mask, MTTask *tasks, size_t capacity)
{
    size_t count = 0;
    for (size_t task = 0; task < MT_TASK_COUNT && count < capacity; ++task) {
        if ((task_mask & mt_task_bit((MTTask)task)) != 0) {
            tasks[count++] = (MTTask)task;
        }
    }
    return count;
}

static const char *mt_training_order_name(MTTrainingOrder order)
{
    switch (order) {
    case MT_TRAINING_UNIFORM_RANDOM:
        return "uniform_random";
    case MT_TRAINING_ROUND_ROBIN:
        return "round_robin";
    case MT_TRAINING_ARC_HEAVY:
        return "arc_heavy";
    case MT_TRAINING_PRIME_LIGHT:
        return "prime_light";
    }
    return "unknown";
}

static int mt_parse_training_order(const char *text, MTTrainingOrder *out_order)
{
    if (text == NULL || out_order == NULL) {
        return 0;
    }
    if (strcmp(text, "uniform_random") == 0 || strcmp(text, "uniform-random") == 0 ||
        strcmp(text, "shuffled") == 0) {
        *out_order = MT_TRAINING_UNIFORM_RANDOM;
        return 1;
    }
    if (strcmp(text, "round_robin") == 0 || strcmp(text, "round-robin") == 0) {
        *out_order = MT_TRAINING_ROUND_ROBIN;
        return 1;
    }
    if (strcmp(text, "arc_heavy") == 0 || strcmp(text, "arc-heavy") == 0) {
        *out_order = MT_TRAINING_ARC_HEAVY;
        return 1;
    }
    if (strcmp(text, "prime_light") == 0 || strcmp(text, "prime-light") == 0) {
        *out_order = MT_TRAINING_PRIME_LIGHT;
        return 1;
    }
    return 0;
}

static void mt_task_set_name(uint32_t task_mask, char *buffer, size_t buffer_size)
{
    size_t written = 0;
    if (buffer_size == 0) {
        return;
    }
    buffer[0] = '\0';
    for (size_t task = 0; task < MT_TASK_COUNT; ++task) {
        if ((task_mask & mt_task_bit((MTTask)task)) == 0) {
            continue;
        }
        int result = snprintf(&buffer[written],
                              buffer_size - written,
                              "%s%s",
                              written == 0 ? "" : "+",
                              mt_task_name((MTTask)task));
        if (result < 0) {
            buffer[0] = '\0';
            return;
        }
        size_t added = (size_t)result;
        if (added >= buffer_size - written) {
            buffer[buffer_size - 1] = '\0';
            return;
        }
        written += added;
    }
}

static uint32_t mt_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double mt_rng_unit(uint32_t *state)
{
    return (double)mt_rng_next(state) / (double)UINT32_MAX;
}

static size_t mt_rng_bounded(uint32_t *state, size_t bound)
{
    if (bound == 0) {
        return 0;
    }
    return (size_t)(((uint64_t)mt_rng_next(state) * (uint64_t)bound) >> 32);
}

static double mt_safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

static int mt_parse_size_text(const char *text, size_t *out_value)
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

static int mt_parse_hidden_text(const char *text, size_t *hidden1, size_t *hidden2)
{
    if (text == NULL || hidden1 == NULL || hidden2 == NULL || text[0] == '\0') {
        return 0;
    }
    char buffer[64];
    int written = snprintf(buffer, sizeof(buffer), "%s", text);
    if (written < 0 || (size_t)written >= sizeof(buffer)) {
        return 0;
    }
    char *sep = strchr(buffer, 'x');
    if (sep == NULL) {
        sep = strchr(buffer, 'X');
    }
    if (sep == NULL) {
        sep = strchr(buffer, ',');
    }
    if (sep == NULL) {
        return 0;
    }
    *sep = '\0';
    size_t first = 0;
    size_t second = 0;
    if (!mt_parse_size_text(buffer, &first) ||
        !mt_parse_size_text(sep + 1, &second)) {
        return 0;
    }
    *hidden1 = first;
    *hidden2 = second;
    return 1;
}

static int mt_parse_u64_text(const char *text, uint64_t *out_value)
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

static int mt_parse_real_text(const char *text, nn_real *out_value)
{
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !isfinite(value) ||
        value <= 0.0) {
        return 0;
    }
    *out_value = (nn_real)value;
    return 1;
}

static int mt_parse_task_set(const char *text, uint32_t *out_mask)
{
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    if (strcmp(text, "all") == 0) {
        uint32_t mask = 0;
        for (size_t task = 0; task < MT_TASK_COUNT; ++task) {
            mask |= mt_task_bit((MTTask)task);
        }
        *out_mask = mask;
        return 1;
    }

    char copy[128];
    size_t length = strlen(text);
    if (length >= sizeof(copy)) {
        return 0;
    }
    memcpy(copy, text, length + 1u);

    uint32_t mask = 0;
    for (char *part = strtok(copy, ",+"); part != NULL; part = strtok(NULL, ",+")) {
        MTTask task;
        if (!mt_task_from_name(part, &task)) {
            return 0;
        }
        mask |= mt_task_bit(task);
    }
    if (mask == 0 || mt_active_task_count(mask) == 0) {
        return 0;
    }

    *out_mask = mask;
    return 1;
}

static const char *mt_feature_envelope_name(const MTConfig *config)
{
    return config->use_task_tags ? "task_tagged_action_rows_v1"
                                 : "untagged_action_rows_v1";
}

static void mt_fill_base_row(const MTConfig *config,
                             nn_real *row,
                             MTTask task,
                             size_t action,
                             size_t candidate_count)
{
    memset(row, 0, MT_FEATURE_COUNT * sizeof(*row));
    row[MT_F_BIAS] = (nn_real)1;
    if (config->use_task_tags) {
        row[MT_F_TASK + (size_t)task] = (nn_real)1;
    }
    row[MT_F_ACTION_INDEX] =
        candidate_count <= 1 ? (nn_real)0 : (nn_real)((double)action / (double)(candidate_count - 1));
    row[MT_F_CANDIDATE_COUNT] = (nn_real)((double)candidate_count / (double)MT_MAX_CANDIDATES);
}

static int mt_grid_cell_index(int x, int y)
{
    return y * MT_GRID_SIZE + x;
}

static int mt_grid_in_bounds(int x, int y)
{
    return x >= 0 && x < MT_GRID_SIZE && y >= 0 && y < MT_GRID_SIZE;
}

static size_t mt_grid_bfs_distances_from_goal(const MTGridState *state, int *dist)
{
    int queue[MT_GRID_CELLS];
    size_t head = 0;
    size_t tail = 0;
    size_t expansions = 0;

    for (size_t i = 0; i < MT_GRID_CELLS; ++i) {
        dist[i] = MT_GRID_INF;
    }

    int goal = mt_grid_cell_index(state->gx, state->gy);
    dist[goal] = 0;
    queue[tail++] = goal;

    while (head < tail) {
        int cell = queue[head++];
        int x = cell % MT_GRID_SIZE;
        int y = cell / MT_GRID_SIZE;
        ++expansions;

        for (size_t action = 0; action < MT_GRID_ACTION_COUNT; ++action) {
            int nx = x + MT_GRID_DX[action];
            int ny = y + MT_GRID_DY[action];
            if (!mt_grid_in_bounds(nx, ny)) {
                continue;
            }
            int next = mt_grid_cell_index(nx, ny);
            if (state->blocked[next] || dist[next] != MT_GRID_INF) {
                continue;
            }
            dist[next] = dist[cell] + 1;
            queue[tail++] = next;
        }
    }

    return expansions;
}

static int mt_generate_grid_state(uint64_t seed, MTGridState *state)
{
    for (uint64_t attempt = 0; attempt < 1000; ++attempt) {
        uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ (attempt * UINT64_C(0x9e3779b9)));
        memset(state, 0, sizeof(*state));

        state->sx = (int)(mt_rng_next(&rng) % MT_GRID_SIZE);
        state->sy = (int)(mt_rng_next(&rng) % MT_GRID_SIZE);
        do {
            state->gx = (int)(mt_rng_next(&rng) % MT_GRID_SIZE);
            state->gy = (int)(mt_rng_next(&rng) % MT_GRID_SIZE);
        } while (state->gx == state->sx && state->gy == state->sy);

        for (int y = 0; y < MT_GRID_SIZE; ++y) {
            for (int x = 0; x < MT_GRID_SIZE; ++x) {
                int cell = mt_grid_cell_index(x, y);
                if ((x == state->sx && y == state->sy) ||
                    (x == state->gx && y == state->gy)) {
                    continue;
                }
                state->blocked[cell] = mt_rng_unit(&rng) < 0.18 ? 1u : 0u;
            }
        }

        int dist[MT_GRID_CELLS];
        mt_grid_bfs_distances_from_goal(state, dist);
        int start = mt_grid_cell_index(state->sx, state->sy);
        if (dist[start] > 0 && dist[start] < MT_GRID_INF) {
            return 1;
        }
    }
    return 0;
}

static MTClaimState mt_generate_claim_state(uint64_t seed)
{
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ UINT64_C(0x9e3779b9));
    MTClaimState state;
    for (size_t i = 0; i < MT_CLAIM_STATE_FEATURES; ++i) {
        state.state[i] = (nn_real)mt_rng_unit(&rng);
    }

    double reliability = (double)state.state[0];
    double evidence = (double)state.state[1];
    double contradiction = (double)state.state[2];
    double age = (double)state.state[3];
    double history = (double)state.state[4];
    double noise = (double)state.state[5];
    double truth_score = 0.45 * reliability + 0.35 * evidence + 0.20 * history -
                         0.30 * contradiction - 0.15 * noise - 0.10 * age;
    int resolves_true = truth_score >= 0.38;
    double uncertainty = fabs(truth_score - 0.38);

    state.utility[0] = resolves_true ? (nn_real)2.0 : (nn_real)-2.5;
    state.utility[1] = resolves_true ? (nn_real)-0.3 : (nn_real)0.3;
    state.utility[2] = (nn_real)(0.95 - 0.6 * uncertainty - 0.25 * age);
    state.utility[3] = resolves_true ? (nn_real)1.1 : (nn_real)-0.2;
    if (contradiction > 0.70 || noise > 0.80) {
        state.utility[2] += (nn_real)0.35;
        state.utility[0] -= (nn_real)0.75;
    }

    state.target = 0;
    for (size_t action = 1; action < MT_CLAIM_ACTION_COUNT; ++action) {
        if (state.utility[action] > state.utility[state.target]) {
            state.target = action;
        }
    }
    return state;
}

static size_t mt_fill_prime_gap_sample(const MTData *data,
                                       const MTConfig *config,
                                       const GapDataset *dataset,
                                       size_t index,
                                       nn_real *candidate_inputs,
                                       size_t *target_index,
                                       double *target_utility)
{
    if (dataset->count == 0) {
        return 0;
    }

    size_t row_index = index % dataset->count;
    uint32_t p = dataset->prime[row_index];
    uint32_t target_gap = dataset->gap[row_index];
    *target_index = gap_to_action(target_gap);
    if (*target_index >= ACTION_COUNT) {
        return 0;
    }
    if (target_utility != NULL) {
        *target_utility = 1.0;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        size_t candidate_gap = action_gap(action);
        uint64_t candidate = (uint64_t)p + (uint64_t)candidate_gap;
        nn_real *row = &candidate_inputs[action * MT_FEATURE_COUNT];
        mt_fill_base_row(config, row, MT_TASK_PRIME_GAP, action, ACTION_COUNT);
        row[MT_F_COMMON + 0] =
            (nn_real)(log((double)p) / data->prime_log_scale);
        row[MT_F_COMMON + 1] =
            (nn_real)((double)candidate_gap / (double)(G_MAX + 2));
        row[MT_F_COMMON + 2] = action == OVERFLOW_ACTION ? (nn_real)1 : (nn_real)0;
        row[MT_F_COMMON + 3] =
            (nn_real)(log1p((double)candidate_gap) / log1p((double)(G_MAX + 2)));
        row[MT_F_COMMON + 4] =
            (nn_real)(log((double)candidate) / data->prime_log_scale);
    }

    return ACTION_COUNT;
}

static size_t mt_fill_grid_sample(const MTConfig *config,
                                  uint64_t seed,
                                  nn_real *candidate_inputs,
                                  size_t *target_index,
                                  double *target_utility)
{
    MTGridState state;
    if (!mt_generate_grid_state(seed, &state)) {
        return 0;
    }

    int dist[MT_GRID_CELLS];
    mt_grid_bfs_distances_from_goal(&state, dist);
    int best_dist = MT_GRID_INF;
    int manhattan_before = abs(state.gx - state.sx) + abs(state.gy - state.sy);
    *target_index = MT_GRID_ACTION_COUNT;

    for (size_t action = 0; action < MT_GRID_ACTION_COUNT; ++action) {
        int nx = state.sx + MT_GRID_DX[action];
        int ny = state.sy + MT_GRID_DY[action];
        int out_of_bounds = !mt_grid_in_bounds(nx, ny);
        int blocked_next = 0;
        int next_dist = MT_GRID_INF;
        int feature_x = state.sx;
        int feature_y = state.sy;

        if (!out_of_bounds) {
            int next_cell = mt_grid_cell_index(nx, ny);
            blocked_next = state.blocked[next_cell] ? 1 : 0;
            feature_x = nx;
            feature_y = ny;
            if (!blocked_next) {
                next_dist = dist[next_cell];
            }
        }

        if (next_dist < MT_GRID_INF &&
            (next_dist < best_dist ||
             (next_dist == best_dist && *target_index == MT_GRID_ACTION_COUNT))) {
            best_dist = next_dist;
            *target_index = action;
        }

        int manhattan_after = abs(state.gx - feature_x) + abs(state.gy - feature_y);
        nn_real *row = &candidate_inputs[action * MT_FEATURE_COUNT];
        mt_fill_base_row(config, row, MT_TASK_GRID_PATH, action, MT_GRID_ACTION_COUNT);
        row[MT_F_COMMON + 0] = (nn_real)((double)state.sx / (double)(MT_GRID_SIZE - 1));
        row[MT_F_COMMON + 1] = (nn_real)((double)state.sy / (double)(MT_GRID_SIZE - 1));
        row[MT_F_COMMON + 2] = (nn_real)((double)state.gx / (double)(MT_GRID_SIZE - 1));
        row[MT_F_COMMON + 3] = (nn_real)((double)state.gy / (double)(MT_GRID_SIZE - 1));
        row[MT_F_COMMON + 4] =
            (nn_real)((double)(state.gx - feature_x) / (double)(MT_GRID_SIZE - 1));
        row[MT_F_COMMON + 5] =
            (nn_real)((double)(state.gy - feature_y) / (double)(MT_GRID_SIZE - 1));
        row[MT_F_COMMON + 6] =
            (nn_real)((double)manhattan_after / (double)(2 * (MT_GRID_SIZE - 1)));
        row[MT_F_COMMON + 7] = (nn_real)((double)(manhattan_before - manhattan_after) / 2.0);
        row[MT_F_STATE + 0] = (nn_real)MT_GRID_DX[action];
        row[MT_F_STATE + 1] = (nn_real)MT_GRID_DY[action];
        row[MT_F_STATE + 2] = out_of_bounds ? (nn_real)1 : (nn_real)0;
        row[MT_F_STATE + 3] = blocked_next ? (nn_real)1 : (nn_real)0;
        for (size_t cell = 0; cell < MT_GRID_CELLS; ++cell) {
            row[MT_F_TASK_SPECIFIC + cell] = state.blocked[cell] ? (nn_real)1 : (nn_real)0;
        }
        for (size_t i = 0; i < MT_GRID_ACTION_COUNT; ++i) {
            row[MT_F_TASK_SPECIFIC + MT_GRID_CELLS + i] = action == i ? (nn_real)1 : (nn_real)0;
        }
    }

    if (*target_index >= MT_GRID_ACTION_COUNT) {
        return 0;
    }
    if (target_utility != NULL) {
        *target_utility = 1.0;
    }
    return MT_GRID_ACTION_COUNT;
}

static size_t mt_fill_claim_sample(const MTConfig *config,
                                   uint64_t seed,
                                   nn_real *candidate_inputs,
                                   size_t *target_index,
                                   double *target_utility)
{
    MTClaimState state = mt_generate_claim_state(seed);
    *target_index = state.target;
    if (target_utility != NULL) {
        *target_utility = (double)state.utility[state.target];
    }

    for (size_t action = 0; action < MT_CLAIM_ACTION_COUNT; ++action) {
        nn_real *row = &candidate_inputs[action * MT_FEATURE_COUNT];
        mt_fill_base_row(config, row, MT_TASK_DELAYED_CLAIM, action, MT_CLAIM_ACTION_COUNT);
        for (size_t i = 0; i < MT_CLAIM_STATE_FEATURES; ++i) {
            row[MT_F_COMMON + i] = state.state[i];
        }
        for (size_t i = 0; i < MT_CLAIM_ACTION_COUNT; ++i) {
            row[MT_F_TASK_SPECIFIC + i] = action == i ? (nn_real)1 : (nn_real)0;
        }
    }

    return MT_CLAIM_ACTION_COUNT;
}

static size_t mt_arc_select_balanced_cell(const uint8_t *output, uint64_t sample)
{
    int want_foreground = (sample & UINT64_C(1)) == 0;
    size_t match_count = 0;
    for (size_t cell = 0; cell < MT_ARC_OUTPUT_CELLS; ++cell) {
        int foreground = output[cell] != 0;
        if (foreground == want_foreground) {
            ++match_count;
        }
    }
    if (match_count == 0) {
        return (size_t)(sample % MT_ARC_OUTPUT_CELLS);
    }

    size_t wanted = (size_t)((sample >> 1) % match_count);
    for (size_t cell = 0; cell < MT_ARC_OUTPUT_CELLS; ++cell) {
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

static size_t mt_fill_arc_sample(const MTConfig *config,
                                 int train,
                                 uint64_t seed_or_index,
                                 nn_real *candidate_inputs,
                                 size_t *target_index,
                                 double *target_utility)
{
    const uint8_t *input = MT_ARC_0520FDE7_TEST_INPUT;
    const uint8_t *output = MT_ARC_0520FDE7_TEST_OUTPUT;
    uint64_t cell_sample = seed_or_index;
    if (train) {
        size_t pair = (size_t)(seed_or_index % MT_ARC_TRAIN_PAIR_COUNT);
        input = MT_ARC_0520FDE7_TRAIN_INPUTS[pair];
        output = MT_ARC_0520FDE7_TRAIN_OUTPUTS[pair];
        cell_sample = seed_or_index / MT_ARC_TRAIN_PAIR_COUNT;
    }

    size_t cell = mt_arc_select_balanced_cell(output, cell_sample);
    size_t output_x = cell % MT_ARC_OUTPUT_WIDTH;
    size_t output_y = cell / MT_ARC_OUTPUT_WIDTH;
    uint8_t target_color = output[cell];
    *target_index = (size_t)target_color;
    if (target_utility != NULL) {
        *target_utility = 1.0;
    }

    for (size_t action = 0; action < MT_ARC_ACTION_COUNT; ++action) {
        nn_real *row = &candidate_inputs[action * MT_FEATURE_COUNT];
        mt_fill_base_row(config, row, MT_TASK_ARC_AGI, action, MT_ARC_ACTION_COUNT);
        row[MT_F_COMMON + 0] =
            (nn_real)((double)output_x / (double)(MT_ARC_OUTPUT_WIDTH - 1));
        row[MT_F_COMMON + 1] =
            (nn_real)((double)output_y / (double)(MT_ARC_OUTPUT_HEIGHT - 1));
        row[MT_F_COMMON + 2] = (nn_real)((double)MT_ARC_INPUT_WIDTH / 30.0);
        row[MT_F_COMMON + 3] = (nn_real)((double)MT_ARC_INPUT_HEIGHT / 30.0);
        row[MT_F_COMMON + 4] =
            (nn_real)((double)MT_ARC_OUTPUT_WIDTH / 30.0);
        row[MT_F_COMMON + 5] =
            (nn_real)((double)MT_ARC_OUTPUT_HEIGHT / 30.0);
        row[MT_F_COMMON + 6] = (nn_real)((double)cell / (double)(MT_ARC_OUTPUT_CELLS - 1));
        row[MT_F_COMMON + 7] = action == 0 ? (nn_real)1 : (nn_real)0;
        row[MT_F_COMMON + 8] = (nn_real)((double)action / 9.0);
        for (size_t i = 0; i < MT_ARC_INPUT_CELLS; ++i) {
            row[MT_F_TASK_SPECIFIC + i] = (nn_real)((double)input[i] / 9.0);
        }
        row[MT_F_TASK_SPECIFIC + MT_ARC_INPUT_CELLS + output_y * MT_ARC_OUTPUT_WIDTH + output_x] =
            (nn_real)1;
        row[MT_F_TASK_SPECIFIC + MT_ARC_INPUT_CELLS + MT_ARC_OUTPUT_CELLS + action] = (nn_real)1;
    }

    return MT_ARC_ACTION_COUNT;
}

static size_t mt_fill_task_sample(const MTData *data,
                                  const MTConfig *config,
                                  MTTask task,
                                  int train,
                                  uint64_t seed_or_index,
                                  nn_real *candidate_inputs,
                                  size_t *target_index,
                                  double *target_utility)
{
    switch (task) {
    case MT_TASK_PRIME_GAP:
        return mt_fill_prime_gap_sample(data,
                                        config,
                                        train ? &data->train : &data->test,
                                        (size_t)seed_or_index,
                                        candidate_inputs,
                                        target_index,
                                        target_utility);
    case MT_TASK_GRID_PATH:
        return mt_fill_grid_sample(config,
                                   train ? 7000u + seed_or_index : 900000u + seed_or_index,
                                   candidate_inputs,
                                   target_index,
                                   target_utility);
    case MT_TASK_DELAYED_CLAIM:
        return mt_fill_claim_sample(config,
                                    train ? 1000u + seed_or_index : 900000u + seed_or_index,
                                    candidate_inputs,
                                    target_index,
                                    target_utility);
    case MT_TASK_ARC_AGI:
        return mt_fill_arc_sample(config,
                                  train,
                                  train ? 5000u + seed_or_index : seed_or_index,
                                  candidate_inputs,
                                  target_index,
                                  target_utility);
    case MT_TASK_COUNT:
        break;
    }
    return 0;
}

static void mt_hidden_text(const MTConfig *config, char *buffer, size_t buffer_size)
{
    if (buffer == NULL || buffer_size == 0) {
        return;
    }
    snprintf(buffer,
             buffer_size,
             "%zux%zu",
             config == NULL ? (size_t)64 : config->hidden1,
             config == NULL ? (size_t)32 : config->hidden2);
}

static NeuralNet *mt_create_model(const MTConfig *config, unsigned int seed)
{
    size_t layers[] = {
        MT_FEATURE_COUNT,
        config == NULL ? (size_t)64 : config->hidden1,
        config == NULL ? (size_t)32 : config->hidden2,
        1,
    };
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    nn_config.seed = seed;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    return nn_create_config(&nn_config);
}

static int mt_prepare_data(MTData *data)
{
    memset(data, 0, sizeof(*data));
    uint32_t train_limit = 100000u;
    uint32_t test_limit = 200000u;
    if (!build_sieve(test_limit + (uint32_t)G_MAX, &data->sieve) ||
        !build_gap_datasets(&data->sieve,
                            train_limit,
                            test_limit,
                            &data->train,
                            &data->validation,
                            &data->test)) {
        free_gap_dataset(&data->train);
        free_gap_dataset(&data->validation);
        free_gap_dataset(&data->test);
        free_sieve(&data->sieve);
        return 0;
    }
    data->prime_log_scale = log((double)test_limit + (double)G_MAX + 2.0);
    return data->train.count != 0 && data->test.count != 0;
}

static void mt_free_data(MTData *data)
{
    free_gap_dataset(&data->train);
    free_gap_dataset(&data->validation);
    free_gap_dataset(&data->test);
    free_sieve(&data->sieve);
    memset(data, 0, sizeof(*data));
}

static int mt_active_task_index(const MTTask *tasks,
                                size_t count,
                                MTTask wanted,
                                size_t *index_out)
{
    if (tasks == NULL || index_out == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (tasks[i] == wanted) {
            *index_out = i;
            return 1;
        }
    }
    return 0;
}

static MTTask mt_choose_nonpreferred_task(const MTTask *tasks,
                                          size_t count,
                                          MTTask preferred,
                                          uint32_t *rng)
{
    if (tasks == NULL || count == 0 || rng == NULL) {
        return MT_TASK_PRIME_GAP;
    }
    size_t alternatives = 0;
    for (size_t i = 0; i < count; ++i) {
        alternatives += tasks[i] != preferred ? 1u : 0u;
    }
    if (alternatives == 0) {
        return preferred;
    }
    size_t wanted = mt_rng_bounded(rng, alternatives);
    for (size_t i = 0; i < count; ++i) {
        if (tasks[i] == preferred) {
            continue;
        }
        if (wanted == 0) {
            return tasks[i];
        }
        --wanted;
    }
    return tasks[0];
}

static MTTask mt_choose_training_task(const MTConfig *config,
                                      const MTTask *active_tasks,
                                      size_t active_count,
                                      size_t global_step,
                                      uint32_t *rng)
{
    if (active_tasks == NULL || active_count == 0 || rng == NULL) {
        return MT_TASK_PRIME_GAP;
    }
    switch ((MTTrainingOrder)(config == NULL ? MT_TRAINING_UNIFORM_RANDOM
                                             : config->training_order)) {
    case MT_TRAINING_ROUND_ROBIN:
        return active_tasks[global_step % active_count];
    case MT_TRAINING_ARC_HEAVY: {
        size_t arc_index = 0;
        if (active_count > 1 &&
            mt_active_task_index(active_tasks,
                                 active_count,
                                 MT_TASK_ARC_AGI,
                                 &arc_index) &&
            mt_rng_unit(rng) < 0.5) {
            return active_tasks[arc_index];
        }
        return mt_choose_nonpreferred_task(active_tasks,
                                           active_count,
                                           MT_TASK_ARC_AGI,
                                           rng);
    }
    case MT_TRAINING_PRIME_LIGHT: {
        size_t prime_index = 0;
        if (active_count > 1 &&
            mt_active_task_index(active_tasks,
                                 active_count,
                                 MT_TASK_PRIME_GAP,
                                 &prime_index)) {
            if (mt_rng_unit(rng) < 0.1) {
                return active_tasks[prime_index];
            }
            return mt_choose_nonpreferred_task(active_tasks,
                                               active_count,
                                               MT_TASK_PRIME_GAP,
                                               rng);
        }
        return active_tasks[mt_rng_bounded(rng, active_count)];
    }
    case MT_TRAINING_UNIFORM_RANDOM:
        break;
    }
    return active_tasks[mt_rng_bounded(rng, active_count)];
}

static int mt_train_mixed(NeuralNet *net,
                          const MTData *data,
                          const MTConfig *config,
                          size_t updates[MT_TASK_COUNT])
{
    uint32_t rng = (uint32_t)(config->seed ^ (config->seed >> 32) ^ UINT64_C(0x31415926));
    MTTask active_tasks[MT_TASK_COUNT];
    size_t active_count = mt_fill_active_tasks(config->task_mask,
                                               active_tasks,
                                               MT_TASK_COUNT);
    nn_real *candidate_inputs =
        calloc((size_t)MT_MAX_CANDIDATES * MT_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (active_count == 0 || candidate_inputs == NULL) {
        free(candidate_inputs);
        return 0;
    }

    memset(updates, 0, MT_TASK_COUNT * sizeof(*updates));
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double losses[MT_TASK_COUNT] = {0.0, 0.0, 0.0, 0.0};
        size_t epoch_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
        for (size_t step = 0; step < config->steps_per_epoch; ++step) {
            size_t global_step = epoch * config->steps_per_epoch + step;
            MTTask task = mt_choose_training_task(config,
                                                  active_tasks,
                                                  active_count,
                                                  global_step,
                                                  &rng);
            uint64_t sample = (uint64_t)mt_rng_next(&rng);
            if (task == MT_TASK_PRIME_GAP && data->train.count != 0) {
                sample %= (uint64_t)data->train.count;
            }

            size_t target_index = 0;
            double ignored_utility = 0.0;
            size_t candidate_count = mt_fill_task_sample(data,
                                                         config,
                                                         task,
                                                         1,
                                                         sample,
                                                         candidate_inputs,
                                                         &target_index,
                                                         &ignored_utility);
            if (candidate_count == 0 || target_index >= candidate_count) {
                free(candidate_inputs);
                return 0;
            }
            nn_real loss = nn_train_action_scorer(net,
                                                  candidate_inputs,
                                                  candidate_count,
                                                  target_index,
                                                  config->learning_rate);
            if (isnan((double)loss)) {
                free(candidate_inputs);
                return 0;
            }
            losses[task] += (double)loss;
            ++epoch_updates[task];
            ++updates[task];
        }

        printf("multitask epoch %zu", epoch + 1);
        for (size_t i = 0; i < active_count; ++i) {
            size_t task = (size_t)active_tasks[i];
            double denom = epoch_updates[task] == 0 ? 1.0 : (double)epoch_updates[task];
            printf(" %s_loss=%.4f", mt_task_name((MTTask)task), losses[task] / denom);
        }
        putchar('\n');
    }

    free(candidate_inputs);
    return 1;
}

static int mt_train_single_task(NeuralNet *net,
                                const MTData *data,
                                const MTConfig *config,
                                MTTask task,
                                size_t updates)
{
    uint32_t rng =
        (uint32_t)(config->seed ^ (config->seed >> 32) ^ (0x9e3779b9u + (uint32_t)task));
    nn_real *candidate_inputs =
        calloc((size_t)MT_MAX_CANDIDATES * MT_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        return 0;
    }

    double loss_sum = 0.0;
    for (size_t step = 0; step < updates; ++step) {
        uint64_t sample = (uint64_t)mt_rng_next(&rng);
        if (task == MT_TASK_PRIME_GAP && data->train.count != 0) {
            sample %= (uint64_t)data->train.count;
        }
        size_t target_index = 0;
        double ignored_utility = 0.0;
        size_t candidate_count = mt_fill_task_sample(data,
                                                     config,
                                                     task,
                                                     1,
                                                     sample,
                                                     candidate_inputs,
                                                     &target_index,
                                                     &ignored_utility);
        if (candidate_count == 0 || target_index >= candidate_count) {
            free(candidate_inputs);
            return 0;
        }
        nn_real loss = nn_train_action_scorer(net,
                                              candidate_inputs,
                                              candidate_count,
                                              target_index,
                                              config->learning_rate);
        if (isnan((double)loss)) {
            free(candidate_inputs);
            return 0;
        }
        loss_sum += (double)loss;
    }

    printf("single_task %s updates=%zu loss=%.4f\n",
           mt_task_name(task),
           updates,
           loss_sum / (updates == 0 ? 1.0 : (double)updates));
    free(candidate_inputs);
    return 1;
}

static MTMetrics mt_evaluate_task(NeuralNet *net,
                                  const MTData *data,
                                  const MTConfig *config,
                                  MTTask task,
                                  size_t samples)
{
    MTMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    nn_real *candidate_inputs =
        calloc((size_t)MT_MAX_CANDIDATES * MT_FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(MT_MAX_CANDIDATES, sizeof(*scores));
    nn_real *probabilities = calloc(MT_MAX_CANDIDATES, sizeof(*probabilities));
    size_t *ranked = calloc(MT_MAX_CANDIDATES, sizeof(*ranked));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL || ranked == NULL) {
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        free(ranked);
        return metrics;
    }

    for (size_t i = 0; i < samples; ++i) {
        size_t target_index = 0;
        double target_utility = 0.0;
        size_t candidate_count = mt_fill_task_sample(data,
                                                     config,
                                                     task,
                                                     0,
                                                     (uint64_t)i,
                                                     candidate_inputs,
                                                     &target_index,
                                                     &target_utility);
        if (candidate_count == 0 || target_index >= candidate_count ||
            !nn_score_action_rows(net, candidate_inputs, candidate_count, scores, probabilities) ||
            !nn_rank_action_scores(scores, candidate_count, ranked)) {
            continue;
        }

        size_t rank = candidate_count;
        for (size_t r = 0; r < candidate_count; ++r) {
            if (ranked[r] == target_index) {
                rank = r;
                break;
            }
        }
        if (rank >= candidate_count) {
            continue;
        }

        ++metrics.samples;
        metrics.top1 += rank == 0 ? 1u : 0u;
        metrics.top5 += rank < 5 ? 1u : 0u;
        metrics.top10 += rank < 10 ? 1u : 0u;
        double confidence = (double)probabilities[ranked[0]];
        int correct = rank == 0;
        size_t bin = (size_t)floor(confidence * (double)MT_CALIBRATION_BINS);
        if (bin >= MT_CALIBRATION_BINS) {
            bin = MT_CALIBRATION_BINS - 1u;
        }
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += correct ? 1.0 : 0.0;
        metrics.confidence_sum += confidence;
        if (correct) {
            metrics.correct_confidence_sum += confidence;
            ++metrics.correct_confidence_count;
        } else {
            metrics.incorrect_confidence_sum += confidence;
            ++metrics.incorrect_confidence_count;
        }
        metrics.nll += -log(mt_safe_probability((double)probabilities[target_index]));
        metrics.mean_rank += (double)rank + 1.0;
        metrics.candidate_count += (double)candidate_count;
        metrics.baseline_steps += task == MT_TASK_DELAYED_CLAIM ? (double)target_index + 1.0
                                                                : (double)candidate_count;
        if (task == MT_TASK_DELAYED_CLAIM) {
            MTClaimState state = mt_generate_claim_state(900000u + (uint64_t)i);
            metrics.utility += (double)state.utility[ranked[0]];
        } else {
            metrics.utility += rank == 0 ? target_utility : 0.0;
        }
    }

    double denom = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= denom;
    metrics.mean_rank /= denom;
    metrics.candidate_count /= denom;
    metrics.utility /= denom;
    metrics.baseline_steps /= denom;
    metrics.confidence_sum /= denom;
    metrics.correct_confidence_sum /=
        metrics.correct_confidence_count == 0 ? 1.0 : (double)metrics.correct_confidence_count;
    metrics.incorrect_confidence_sum /=
        metrics.incorrect_confidence_count == 0 ? 1.0 : (double)metrics.incorrect_confidence_count;
    for (size_t bin = 0; bin < MT_CALIBRATION_BINS; ++bin) {
        if (metrics.bin_count[bin] == 0) {
            continue;
        }
        double count = (double)metrics.bin_count[bin];
        metrics.bin_confidence[bin] /= count;
        metrics.bin_accuracy[bin] /= count;
        metrics.ece +=
            (count / denom) * fabs(metrics.bin_confidence[bin] - metrics.bin_accuracy[bin]);
    }

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free(ranked);
    return metrics;
}

static void mt_print_reliability_bins_json(const MTMetrics *metrics)
{
    int first = 1;
    fputs(",\"reliability_bins\":[", stdout);
    if (metrics != NULL) {
        for (size_t bin = 0; bin < MT_CALIBRATION_BINS; ++bin) {
            if (metrics->bin_count[bin] == 0) {
                continue;
            }
            if (!first) {
                fputc(',', stdout);
            }
            first = 0;
            printf("{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
                   "\"confidence\":%.8f,\"accuracy\":%.8f}",
                   (double)bin / (double)MT_CALIBRATION_BINS,
                   (double)(bin + 1u) / (double)MT_CALIBRATION_BINS,
                   metrics->bin_count[bin],
                   metrics->bin_confidence[bin],
                   metrics->bin_accuracy[bin]);
        }
    }
    fputc(']', stdout);
}

static void mt_print_metrics_json(const NeuralNet *net,
                                  const char *model_kind,
                                  const char *training_order,
                                  MTTask task,
                                  size_t train_updates,
                                  const MTConfig *config,
                                  MTMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    char task_set[128];
    char eval_task_set[128];
    char hidden[32];
    mt_task_set_name(config->task_mask, task_set, sizeof(task_set));
    mt_task_set_name(config->eval_task_mask, eval_task_set, sizeof(eval_task_set));
    mt_hidden_text(config, hidden, sizeof(hidden));
    printf("{\"schema\":\"crlp.multitask.summary.v1\","
           "\"task\":\"multi_domain\","
           "\"domain\":\"%s\","
           "\"task_set\":\"%s\","
           "\"eval_task_set\":\"%s\","
           "\"model_kind\":\"%s\","
           "\"training_order\":\"%s\","
           "\"feature_envelope\":\"%s\","
           "\"train_updates\":%zu,"
           "\"epochs\":%zu,"
           "\"steps_per_epoch\":%zu,"
           "\"hidden\":\"%s\","
           "\"parameters\":%zu,"
           "\"forward_flops\":%zu,"
           "\"learning_rate\":%.8f,"
           "\"seed\":%llu,"
           "\"sample_count\":%zu,"
           "\"candidate_count\":%.8f,"
           "\"top1\":%.8f,"
           "\"top5\":%.8f,"
           "\"top10\":%.8f,"
           "\"nll\":%.8f,"
           "\"ece\":%.8f,"
           "\"confidence_mean\":%.8f,"
           "\"correct_confidence_mean\":%.8f,"
           "\"incorrect_confidence_mean\":%.8f,"
           "\"mean_rank\":%.8f,"
           "\"decision_utility\":%.8f,"
           "\"baseline_steps\":%.8f",
           mt_task_name(task),
           task_set,
           eval_task_set,
           model_kind,
           training_order,
           mt_feature_envelope_name(config),
           train_updates,
           config->epochs,
           config->steps_per_epoch,
           hidden,
           net == NULL ? (size_t)0 : nn_parameter_count(net),
           net == NULL ? (size_t)0 : nn_forward_flop_count(net),
           (double)config->learning_rate,
           (unsigned long long)config->seed,
           metrics.samples,
           metrics.candidate_count,
           (double)metrics.top1 / samples,
           (double)metrics.top5 / samples,
           (double)metrics.top10 / samples,
           metrics.nll,
           metrics.ece,
           metrics.confidence_sum,
           metrics.correct_confidence_sum,
           metrics.incorrect_confidence_sum,
           metrics.mean_rank,
           metrics.utility,
           metrics.baseline_steps);
    mt_print_reliability_bins_json(&metrics);
    printf("}\n");
}

static void mt_print_metrics_line(const char *model_kind, MTTask task, MTMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    printf("[%s %s] top1=%.2f%% top5=%.2f%% top10=%.2f%% nll=%.4f "
           "ece=%.4f mean_rank=%.2f candidates=%.2f utility=%.4f samples=%zu\n",
           model_kind,
           mt_task_name(task),
           100.0 * (double)metrics.top1 / samples,
           100.0 * (double)metrics.top5 / samples,
           100.0 * (double)metrics.top10 / samples,
           metrics.nll,
           metrics.ece,
           metrics.mean_rank,
           metrics.candidate_count,
           metrics.utility,
           metrics.samples);
}

static void mt_print_usage(FILE *stream)
{
    fputs("usage: multitask_domain [--epochs count] [--steps-per-epoch count] "
          "[--eval-samples count] [--learning-rate rate] [--seed seed] "
          "[--hidden h1xh2] "
          "[--training-order uniform_random|round_robin|arc_heavy|prime_light] "
          "[--tasks prime,grid,claim,arc|all] [--eval-tasks prime,grid,claim,arc|all] "
          "[--no-task-tags] [--no-task-baselines]\n",
          stream);
}

static int mt_parse_config(int argc, char **argv, MTConfig *config)
{
    config->epochs = MT_DEFAULT_EPOCHS;
    config->steps_per_epoch = MT_DEFAULT_STEPS_PER_EPOCH;
    config->eval_samples = MT_DEFAULT_EVAL_SAMPLES;
    config->seed = 20240516u;
    config->learning_rate = (nn_real)0.006;
    config->hidden1 = 64;
    config->hidden2 = 32;
    config->run_task_baselines = 1;
    config->use_task_tags = 1;
    config->task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                        mt_task_bit(MT_TASK_GRID_PATH) |
                        mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                        mt_task_bit(MT_TASK_ARC_AGI);
    config->eval_task_mask = 0;
    config->training_order = MT_TRAINING_UNIFORM_RANDOM;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!mt_parse_size_text(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--steps-per-epoch") == 0 && i + 1 < argc) {
            if (!mt_parse_size_text(argv[++i], &config->steps_per_epoch)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--eval-samples") == 0 && i + 1 < argc) {
            if (!mt_parse_size_text(argv[++i], &config->eval_samples)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            if (!mt_parse_real_text(argv[++i], &config->learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            if (!mt_parse_hidden_text(argv[++i], &config->hidden1, &config->hidden2)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--training-order") == 0 && i + 1 < argc) {
            MTTrainingOrder order = MT_TRAINING_UNIFORM_RANDOM;
            if (!mt_parse_training_order(argv[++i], &order)) {
                return 0;
            }
            config->training_order = order;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!mt_parse_u64_text(argv[++i], &config->seed)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--tasks") == 0 && i + 1 < argc) {
            if (!mt_parse_task_set(argv[++i], &config->task_mask)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--eval-tasks") == 0 && i + 1 < argc) {
            if (!mt_parse_task_set(argv[++i], &config->eval_task_mask)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--no-task-tags") == 0) {
            config->use_task_tags = 0;
        } else if (strcmp(argv[i], "--no-task-baselines") == 0) {
            config->run_task_baselines = 0;
        } else {
            return 0;
        }
    }
    if (config->eval_task_mask == 0) {
        config->eval_task_mask = config->task_mask;
    }
    return 1;
}

int main(int argc, char **argv)
{
    MTConfig config;
    if (!mt_parse_config(argc, argv, &config)) {
        mt_print_usage(stderr);
        return EXIT_FAILURE;
    }

    MTData data;
    if (!mt_prepare_data(&data)) {
        fputs("failed to prepare multitask datasets\n", stderr);
        return EXIT_FAILURE;
    }

    NeuralNet *shared = mt_create_model(&config, (unsigned int)(config.seed & 0xffffffffu));
    if (shared == NULL) {
        mt_free_data(&data);
        return EXIT_FAILURE;
    }

    char task_set[128];
    char eval_task_set[128];
    char hidden[32];
    mt_task_set_name(config.task_mask, task_set, sizeof(task_set));
    mt_task_set_name(config.eval_task_mask, eval_task_set, sizeof(eval_task_set));
    mt_hidden_text(&config, hidden, sizeof(hidden));
    printf("[multitask_domain] feature_envelope=%s "
           "task_set=%s eval_task_set=%s epochs=%zu steps_per_epoch=%zu "
           "eval_samples=%zu hidden=%s training_order=%s lr=%.6f seed=%llu\n",
           mt_feature_envelope_name(&config),
           task_set,
           eval_task_set,
           config.epochs,
           config.steps_per_epoch,
           config.eval_samples,
           hidden,
           mt_training_order_name((MTTrainingOrder)config.training_order),
           (double)config.learning_rate,
           (unsigned long long)config.seed);

    size_t shared_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
    if (!mt_train_mixed(shared, &data, &config, shared_updates)) {
        nn_free(shared);
        mt_free_data(&data);
        return EXIT_FAILURE;
    }

    MTTask active_tasks[MT_TASK_COUNT];
    size_t active_count = mt_fill_active_tasks(config.task_mask,
                                               active_tasks,
                                               MT_TASK_COUNT);
    MTTask eval_tasks[MT_TASK_COUNT];
    size_t eval_count = mt_fill_active_tasks(config.eval_task_mask,
                                             eval_tasks,
                                             MT_TASK_COUNT);
    for (size_t task_index = 0; task_index < eval_count; ++task_index) {
        MTTask task = eval_tasks[task_index];
        MTMetrics metrics = mt_evaluate_task(shared, &data, &config, task, config.eval_samples);
        mt_print_metrics_json(shared,
                              "shared",
                              mt_training_order_name((MTTrainingOrder)config.training_order),
                              task,
                              shared_updates[task],
                              &config,
                              metrics);
        mt_print_metrics_line("shared", task, metrics);
    }

    if (config.run_task_baselines) {
        size_t baseline_updates =
            (config.epochs * config.steps_per_epoch) / active_count;
        if (baseline_updates == 0) {
            baseline_updates = 1;
        }
        for (size_t task_index = 0; task_index < active_count; ++task_index) {
            MTTask task = active_tasks[task_index];
            NeuralNet *single =
                mt_create_model(&config,
                                (unsigned int)((config.seed + 101u + task_index) &
                                               0xffffffffu));
            if (single == NULL ||
                !mt_train_single_task(single, &data, &config, task, baseline_updates)) {
                nn_free(single);
                nn_free(shared);
                mt_free_data(&data);
                return EXIT_FAILURE;
            }
            MTMetrics metrics = mt_evaluate_task(single, &data, &config, task, config.eval_samples);
            mt_print_metrics_json(single,
                                  "single_task",
                                  "domain_only_random",
                                  task,
                                  baseline_updates,
                                  &config,
                                  metrics);
            mt_print_metrics_line("single_task", task, metrics);
            nn_free(single);
        }
    }

    nn_free(shared);
    mt_free_data(&data);
    return EXIT_SUCCESS;
}
