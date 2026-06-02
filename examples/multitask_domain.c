#include "nn.h"
#include "../src/prime_gap/internal.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef enum {
    MT_TASK_PRIME_GAP = 0,
    MT_TASK_GRID_PATH = 1,
    MT_TASK_DELAYED_CLAIM = 2,
    MT_TASK_ARC_AGI = 3,
    MT_TASK_GO = 4,
    MT_TASK_COUNT = 5,
} MTTask;

typedef enum {
    MT_ARC_EVAL_0520FDE7 = 0,
    MT_ARC_EVAL_OOD_SAME_ENVELOPE = 1,
} MTArcEvalSuite;

typedef enum {
    MT_ARC_OOD_UNION = 0,
    MT_ARC_OOD_XOR = 1,
    MT_ARC_OOD_LEFT_ONLY = 2,
    MT_ARC_OOD_RIGHT_ONLY = 3,
    MT_ARC_OOD_LEFT_PANEL = 4,
    MT_ARC_OOD_RIGHT_PANEL = 5,
    MT_ARC_OOD_NEITHER = 6,
    MT_ARC_OOD_NOT_LEFT = 7,
    MT_ARC_OOD_NOT_RIGHT = 8,
    MT_ARC_OOD_LEFT_IMPLIES_RIGHT = 9,
    MT_ARC_OOD_RIGHT_IMPLIES_LEFT = 10,
    MT_ARC_OOD_ROW_ANY_INTERSECTION = 11,
} MTArcOODFamily;

typedef struct {
    const char *name;
    const char *family_name;
    MTArcOODFamily family;
    uint16_t left_mask;
    uint16_t right_mask;
} MTArcOODTask;

typedef struct MTReportRows MTReportRows;

enum {
    MT_FEATURE_COUNT = NN_SCORER_ENVELOPE_FEATURE_COUNT,
    MT_MAX_CANDIDATES = ACTION_COUNT,
    MT_CALIBRATION_BINS = 10,
    MT_GRID_SIZE = 7,
    MT_GRID_CELLS = MT_GRID_SIZE * MT_GRID_SIZE,
    MT_GRID_ACTION_COUNT = 4,
    MT_GRID_INF = 1000000,
    MT_GO_SIZE = 9,
    MT_GO_CELLS = MT_GO_SIZE * MT_GO_SIZE,
    MT_GO_ACTION_COUNT = MT_GO_CELLS + 1,
    MT_GO_MAX_MOVES = 150,
    MT_CLAIM_STATE_FEATURES = 6,
    MT_CLAIM_ACTION_COUNT = 4,
    MT_ARC_INPUT_WIDTH = 7,
    MT_ARC_INPUT_HEIGHT = 3,
    MT_ARC_INPUT_CELLS = MT_ARC_INPUT_WIDTH * MT_ARC_INPUT_HEIGHT,
    MT_ARC_OUTPUT_WIDTH = 3,
    MT_ARC_OUTPUT_HEIGHT = 3,
    MT_ARC_OUTPUT_CELLS = MT_ARC_OUTPUT_WIDTH * MT_ARC_OUTPUT_HEIGHT,
    MT_ARC_PANEL_WIDTH = 3,
    MT_ARC_PANEL_HEIGHT = 3,
    MT_ARC_PANEL_CELLS = MT_ARC_PANEL_WIDTH * MT_ARC_PANEL_HEIGHT,
    MT_PRIME_RESIDUE_FEATURES = FEATURE_MOD_COUNT,
    MT_ARC_SEPARATOR_COLOR = 5,
    MT_ARC_INPUT_COLOR = 1,
    MT_ARC_TARGET_COLOR = 2,
    MT_ARC_ACTION_COUNT = 10,
    MT_ARC_TRAIN_PAIR_COUNT = 3,
    MT_ARC_OOD_TASK_COUNT = 12,
    MT_F_BIAS = NN_SCORER_ENVELOPE_F_BIAS,
    MT_F_ACTION_INDEX = NN_SCORER_ENVELOPE_F_CANDIDATE_INDEX,
    MT_F_CANDIDATE_COUNT = NN_SCORER_ENVELOPE_F_CANDIDATE_COUNT,
    MT_F_COMMON = NN_SCORER_ENVELOPE_F_ACTION_COMMON,
    MT_F_STATE = NN_SCORER_ENVELOPE_F_ACTION_STATE,
    MT_F_TASK_SPECIFIC = NN_SCORER_ENVELOPE_F_ROLE_PAYLOAD,
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
static const MTArcOODTask MT_ARC_OOD_TASKS[MT_ARC_OOD_TASK_COUNT] = {
    {"union_checker", "union", MT_ARC_OOD_UNION, 0x155u, 0x0bau},
    {"xor_offset", "xor", MT_ARC_OOD_XOR, 0x0d3u, 0x18eu},
    {"left_only_bands", "left_only", MT_ARC_OOD_LEFT_ONLY, 0x1c7u, 0x092u},
    {"right_only_floor", "right_only", MT_ARC_OOD_RIGHT_ONLY, 0x049u, 0x1f8u},
    {"copy_left_sparse", "copy_left", MT_ARC_OOD_LEFT_PANEL, 0x0adu, 0x151u},
    {"copy_right_sparse", "copy_right", MT_ARC_OOD_RIGHT_PANEL, 0x056u, 0x12bu},
    {"neither_diag", "neither", MT_ARC_OOD_NEITHER, 0x111u, 0x044u},
    {"not_left_motif", "not_left", MT_ARC_OOD_NOT_LEFT, 0x0d6u, 0x12cu},
    {"not_right_motif", "not_right", MT_ARC_OOD_NOT_RIGHT, 0x0a3u, 0x169u},
    {"left_implies_right", "left_implies_right", MT_ARC_OOD_LEFT_IMPLIES_RIGHT, 0x0f3u, 0x05au},
    {"right_implies_left", "right_implies_left", MT_ARC_OOD_RIGHT_IMPLIES_LEFT, 0x0a6u, 0x1b2u},
    {"row_any_intersection", "row_any_intersection", MT_ARC_OOD_ROW_ANY_INTERSECTION, 0x129u, 0x083u},
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
    int arc_eval_suite;
    int arc_ood_task;
    int report_arc_ood_tasks;
    int run_task_head_model;
    int adapter_heads_report;
    int tiny_net_plateau_audit;
    int tiny_net_resume;
    size_t tiny_net_shard_index;
    size_t tiny_net_shard_count;
    int quiet;
    const char *output_dir;
    const char *report_seeds;
    const char *tiny_net_capacities;
    const char *tiny_net_epoch_grid;
    const char *roadmap_item;
    const char *adapter_eval_suite;
    double transfer_regression_margin;
    double plateau_threshold;
    size_t plateau_window;
    FILE *json_output;
    MTReportRows *report_rows;
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

typedef struct {
    size_t cells;
    size_t correct_cells;
    int grid_exact;
    uint64_t prediction_hash;
    double nll;
    uint8_t target[MT_ARC_OUTPUT_CELLS];
    uint8_t predicted[MT_ARC_OUTPUT_CELLS];
    double confidence[MT_ARC_OUTPUT_CELLS];
    double target_probability[MT_ARC_OUTPUT_CELLS];
    size_t target_color_count[MT_ARC_ACTION_COUNT];
    size_t predicted_color_count[MT_ARC_ACTION_COUNT];
    size_t correct_target_color_count[MT_ARC_ACTION_COUNT];
} MTArcFixedDiagnostics;

typedef struct {
    size_t input_count;
    size_t hidden1;
    size_t hidden2;
    size_t candidate_capacity;
    nn_real *w1;
    nn_real *b1;
    nn_real *w2;
    nn_real *b2;
    nn_real *head_w;
    nn_real *head_b;
    nn_real *a1;
    nn_real *a2;
    nn_real *scores;
    nn_real *probabilities;
    nn_real *delta1;
    nn_real *delta2;
    nn_real *grad_w1;
    nn_real *grad_b1;
    nn_real *grad_w2;
    nn_real *grad_b2;
    nn_real *grad_head_w;
} MTTaskHeadedScorer;

typedef enum {
    MT_SCORER_NN = 0,
    MT_SCORER_TASK_HEADS = 1,
} MTScorerKind;

typedef struct {
    MTScorerKind kind;
    NeuralNet *net;
    MTTaskHeadedScorer *task_heads;
} MTScorer;

typedef struct {
    const char *eval_suite;
    const char *domain;
    const char *model_kind;
    char hidden[32];
    uint64_t seed;
    size_t epochs;
    double top1;
    double nll;
    double ece;
    double decision_utility;
    double train_updates;
    double parameters;
    double forward_flops;
} MTReportRow;

struct MTReportRows {
    MTReportRow *items;
    size_t count;
    size_t capacity;
};

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
    case MT_TASK_GO:
        return "go";
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
    if (strcmp(name, "go") == 0 || strcmp(name, "go_position") == 0) {
        *out_task = MT_TASK_GO;
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

static FILE *mt_json_output(const MTConfig *config)
{
    return config != NULL && config->json_output != NULL ? config->json_output : stdout;
}

static void mt_json_write_string(FILE *out, const char *text)
{
    const unsigned char *cursor = (const unsigned char *)(text == NULL ? "" : text);
    fputc('"', out);
    while (*cursor != '\0') {
        unsigned char ch = *cursor++;
        switch (ch) {
        case '"':
            fputs("\\\"", out);
            break;
        case '\\':
            fputs("\\\\", out);
            break;
        case '\b':
            fputs("\\b", out);
            break;
        case '\f':
            fputs("\\f", out);
            break;
        case '\n':
            fputs("\\n", out);
            break;
        case '\r':
            fputs("\\r", out);
            break;
        case '\t':
            fputs("\\t", out);
            break;
        default:
            if (ch < 0x20U) {
                fprintf(out, "\\u%04x", (unsigned int)ch);
            } else {
                fputc((int)ch, out);
            }
            break;
        }
    }
    fputc('"', out);
}

static void mt_print_report_metadata_json(FILE *out, const MTConfig *config)
{
    if (config == NULL) {
        return;
    }
    if (config->roadmap_item != NULL && config->roadmap_item[0] != '\0') {
        fputs(",\"roadmap_item\":", out);
        mt_json_write_string(out, config->roadmap_item);
    }
    if (config->adapter_eval_suite != NULL && config->adapter_eval_suite[0] != '\0') {
        fputs(",\"adapter_eval_suite\":", out);
        mt_json_write_string(out, config->adapter_eval_suite);
    }
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

static int mt_parse_shard_text(const char *text, size_t *index, size_t *count)
{
    char buffer[64];
    char *sep;
    size_t parsed_index = 0U;
    size_t parsed_count = 0U;
    int written;
    if (text == NULL || index == NULL || count == NULL || text[0] == '\0') {
        return 0;
    }
    written = snprintf(buffer, sizeof(buffer), "%s", text);
    if (written < 0 || (size_t)written >= sizeof(buffer)) {
        return 0;
    }
    sep = strchr(buffer, '/');
    if (sep == NULL) {
        return 0;
    }
    *sep = '\0';
    if (!mt_parse_size_text(buffer, &parsed_index) ||
        !mt_parse_size_text(sep + 1, &parsed_count) ||
        parsed_index > parsed_count) {
        return 0;
    }
    *index = parsed_index;
    *count = parsed_count;
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
    return config->use_task_tags ? nn_scorer_envelope_feature_set()
                                 : "untagged_action_rows_v1";
}

static NNScorerDomain mt_scorer_domain(MTTask task)
{
    switch (task) {
    case MT_TASK_PRIME_GAP:
        return NN_SCORER_DOMAIN_PRIME_GAP_ACTION;
    case MT_TASK_GRID_PATH:
        return NN_SCORER_DOMAIN_GRID_PATH_ACTION;
    case MT_TASK_DELAYED_CLAIM:
        return NN_SCORER_DOMAIN_DELAYED_CLAIM_ACTION;
    case MT_TASK_ARC_AGI:
        return NN_SCORER_DOMAIN_ARC_AGI_ACTION;
    case MT_TASK_GO:
        return NN_SCORER_DOMAIN_GO_POSITION_ACTION;
    case MT_TASK_COUNT:
        break;
    }
    return NN_SCORER_DOMAIN_PRIME_GAP_ACTION;
}

static void mt_fill_base_row(const MTConfig *config,
                             nn_real *row,
                             MTTask task,
                             size_t action,
                             size_t candidate_count)
{
    NNScorerEnvelopeRow spec = {
        .role = NN_SCORER_ROLE_ACTION,
        .domain = mt_scorer_domain(task),
        .candidate_index = action,
        .candidate_count = candidate_count,
        .candidate_count_scale = MT_MAX_CANDIDATES,
        .include_role_tag = config->use_task_tags,
        .include_domain_tag = config->use_task_tags,
    };
    (void)nn_scorer_envelope_fill_base(row, MT_FEATURE_COUNT, &spec);
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
        /* Match the dedicated prime-gap scorer's wheel bits; global gap-frequency
           cues are too weak alone. */
        if (action == OVERFLOW_ACTION) {
            for (size_t i = 0; i < MT_PRIME_RESIDUE_FEATURES; ++i) {
                row[MT_F_TASK_SPECIFIC + i] = (nn_real)0;
            }
            continue;
        }
        for (size_t i = 0; i < MT_PRIME_RESIDUE_FEATURES; ++i) {
            uint32_t mod = FEATURE_MODS[i];
            row[MT_F_TASK_SPECIFIC + i] =
                candidate > mod && candidate % mod == 0 ? (nn_real)1 : (nn_real)0;
        }
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

static const char *mt_arc_eval_suite_name(int suite)
{
    switch ((MTArcEvalSuite)suite) {
    case MT_ARC_EVAL_0520FDE7:
        return "0520fde7";
    case MT_ARC_EVAL_OOD_SAME_ENVELOPE:
        return "ood_same_envelope_v1";
    }
    return "unknown";
}

static int mt_parse_arc_eval_suite(const char *text, int *out_suite)
{
    if (text == NULL || out_suite == NULL) {
        return 0;
    }
    if (strcmp(text, "0520fde7") == 0 || strcmp(text, "fixed") == 0 ||
        strcmp(text, "id") == 0 || strcmp(text, "in_distribution") == 0 ||
        strcmp(text, "in-distribution") == 0) {
        *out_suite = MT_ARC_EVAL_0520FDE7;
        return 1;
    }
    if (strcmp(text, "ood") == 0 || strcmp(text, "ood_same_envelope") == 0 ||
        strcmp(text, "ood-same-envelope") == 0 ||
        strcmp(text, "ood_same_envelope_v1") == 0) {
        *out_suite = MT_ARC_EVAL_OOD_SAME_ENVELOPE;
        return 1;
    }
    return 0;
}

static int mt_parse_arc_ood_task(const char *text, int *out_task)
{
    if (text == NULL || out_task == NULL) {
        return 0;
    }
    if (strcmp(text, "all") == 0 || strcmp(text, "*") == 0) {
        *out_task = -1;
        return 1;
    }
    errno = 0;
    char *end = NULL;
    unsigned long parsed = strtoul(text, &end, 10);
    if (end != text && *end == '\0' && errno != ERANGE) {
        if (parsed >= MT_ARC_OOD_TASK_COUNT) {
            return 0;
        }
        *out_task = (int)parsed;
        return 1;
    }
    for (size_t i = 0; i < MT_ARC_OOD_TASK_COUNT; ++i) {
        if (strcmp(text, MT_ARC_OOD_TASKS[i].name) == 0 ||
            strcmp(text, MT_ARC_OOD_TASKS[i].family_name) == 0) {
            *out_task = (int)i;
            return 1;
        }
    }
    return 0;
}

static int mt_arc_mask_cell(uint16_t mask, size_t cell)
{
    return ((mask >> cell) & 1u) != 0u;
}

static int mt_arc_ood_rule_value(MTArcOODFamily family,
                                 uint16_t left_mask,
                                 uint16_t right_mask,
                                 size_t cell)
{
    int left = mt_arc_mask_cell(left_mask, cell);
    int right = mt_arc_mask_cell(right_mask, cell);
    switch (family) {
    case MT_ARC_OOD_UNION:
        return left || right;
    case MT_ARC_OOD_XOR:
        return left != right;
    case MT_ARC_OOD_LEFT_ONLY:
        return left && !right;
    case MT_ARC_OOD_RIGHT_ONLY:
        return right && !left;
    case MT_ARC_OOD_LEFT_PANEL:
        return left;
    case MT_ARC_OOD_RIGHT_PANEL:
        return right;
    case MT_ARC_OOD_NEITHER:
        return !left && !right;
    case MT_ARC_OOD_NOT_LEFT:
        return !left;
    case MT_ARC_OOD_NOT_RIGHT:
        return !right;
    case MT_ARC_OOD_LEFT_IMPLIES_RIGHT:
        return !left || right;
    case MT_ARC_OOD_RIGHT_IMPLIES_LEFT:
        return !right || left;
    case MT_ARC_OOD_ROW_ANY_INTERSECTION: {
        size_t row = cell / MT_ARC_PANEL_WIDTH;
        uint16_t row_mask = (uint16_t)(0x7u << (row * MT_ARC_PANEL_WIDTH));
        return (left_mask & right_mask & row_mask) != 0u;
    }
    }
    return 0;
}

static void mt_arc_fill_input_from_masks(uint16_t left_mask, uint16_t right_mask, uint8_t *input)
{
    for (size_t y = 0; y < MT_ARC_PANEL_HEIGHT; ++y) {
        for (size_t x = 0; x < MT_ARC_PANEL_WIDTH; ++x) {
            size_t panel_cell = y * MT_ARC_PANEL_WIDTH + x;
            input[y * MT_ARC_INPUT_WIDTH + x] =
                mt_arc_mask_cell(left_mask, panel_cell) ? MT_ARC_INPUT_COLOR : 0u;
            input[y * MT_ARC_INPUT_WIDTH + MT_ARC_PANEL_WIDTH] = MT_ARC_SEPARATOR_COLOR;
            input[y * MT_ARC_INPUT_WIDTH + MT_ARC_PANEL_WIDTH + 1u + x] =
                mt_arc_mask_cell(right_mask, panel_cell) ? MT_ARC_INPUT_COLOR : 0u;
        }
    }
}

static void mt_arc_generate_ood_task(size_t task_index, uint8_t *input, uint8_t *output)
{
    const MTArcOODTask *task = &MT_ARC_OOD_TASKS[task_index % MT_ARC_OOD_TASK_COUNT];
    mt_arc_fill_input_from_masks(task->left_mask, task->right_mask, input);
    for (size_t cell = 0; cell < MT_ARC_OUTPUT_CELLS; ++cell) {
        output[cell] = mt_arc_ood_rule_value(task->family,
                                             task->left_mask,
                                             task->right_mask,
                                             cell)
                           ? MT_ARC_TARGET_COLOR
                           : 0u;
    }
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

static size_t mt_fill_arc_cell_rows(const MTConfig *config,
                                    const uint8_t *input,
                                    const uint8_t *output,
                                    size_t cell,
                                    nn_real *candidate_inputs,
                                    size_t *target_index,
                                    double *target_utility)
{
    if (input == NULL || output == NULL || candidate_inputs == NULL ||
        target_index == NULL || cell >= MT_ARC_OUTPUT_CELLS) {
        return 0;
    }
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

static size_t mt_fill_arc_sample(const MTConfig *config,
                                 int train,
                                 uint64_t seed_or_index,
                                 nn_real *candidate_inputs,
                                 size_t *target_index,
                                 double *target_utility)
{
    const uint8_t *input = MT_ARC_0520FDE7_TEST_INPUT;
    const uint8_t *output = MT_ARC_0520FDE7_TEST_OUTPUT;
    uint8_t ood_input[MT_ARC_INPUT_CELLS];
    uint8_t ood_output[MT_ARC_OUTPUT_CELLS];
    uint64_t cell_sample = seed_or_index;
    if (train) {
        size_t pair = (size_t)(seed_or_index % MT_ARC_TRAIN_PAIR_COUNT);
        input = MT_ARC_0520FDE7_TRAIN_INPUTS[pair];
        output = MT_ARC_0520FDE7_TRAIN_OUTPUTS[pair];
        cell_sample = seed_or_index / MT_ARC_TRAIN_PAIR_COUNT;
    } else if (config != NULL &&
               config->arc_eval_suite == MT_ARC_EVAL_OOD_SAME_ENVELOPE) {
        size_t task_index = 0;
        if (config->arc_ood_task >= 0) {
            task_index = (size_t)config->arc_ood_task;
            cell_sample = seed_or_index;
        } else {
            task_index = (size_t)(seed_or_index % MT_ARC_OOD_TASK_COUNT);
            cell_sample = seed_or_index / MT_ARC_OOD_TASK_COUNT;
        }
        mt_arc_generate_ood_task(task_index, ood_input, ood_output);
        input = ood_input;
        output = ood_output;
    }

    size_t cell = mt_arc_select_balanced_cell(output, cell_sample);
    return mt_fill_arc_cell_rows(config,
                                 input,
                                 output,
                                 cell,
                                 candidate_inputs,
                                 target_index,
                                 target_utility);
}

/* ── Go position task ────────────────────────────────────────────────── */

#include "go_rules.h"

static size_t mt_fill_go_sample(const MTConfig *config,
                                 uint64_t seed,
                                 nn_real *candidate_inputs,
                                 size_t *target_index,
                                 double *target_utility)
{
    /* Generate a random board position by playing random legal moves */
    GoBoard b; go_init(&b);
    srand((unsigned)seed);
    int n_placed = (rand() % 18) + 3;
    for (int p = 0; p < n_placed && !go_over(&b); p++) {
        int lg[GO_MM], nl = 0;
        for (int m = 0; m < GO_MM; m++) if (go_legal(&b, m)) lg[nl++] = m;
        if (!nl) break;
        go_move(&b, lg[rand() % nl]);
    }
    /* Retry if the random position has no legal moves (rare). */
    for (int retry = 0; retry < 20; retry++) {
        int nlc = 0;
        for (int m = 0; m < GO_MM; m++) if (go_legal(&b, m)) nlc++;
        if (nlc > 0 && !go_over(&b)) break;
        /* Regenerate with a different seed */
        go_init(&b);
        srand((unsigned)(seed + (uint64_t)retry * 9973u));
        int np = (rand() % 18) + 3;
        for (int p = 0; p < np && !go_over(&b); p++) {
            int lg[GO_MM], nll = 0;
            for (int m = 0; m < GO_MM; m++) if (go_legal(&b, m)) lg[nll++] = m;
            if (!nll) break;
            go_move(&b, lg[rand() % nll]);
        }
    }
    int nlc = 0;
    for (int m = 0; m < GO_MM; m++) if (go_legal(&b, m)) nlc++;
    if (!nlc || go_over(&b)) return 0;

    /* Deterministic replay for each legal move */
    float best = -9999.0f, sc[GO_MM];
    for (int m = 0; m < GO_MM; m++) {
        if (go_legal(&b, m)) {
            sc[m] = go_replay(&b, m, (unsigned)seed + (unsigned)m);
            if (sc[m] > best) best = sc[m];
        } else sc[m] = -9999.0f;
    }

    *target_index = (size_t)-1;
    *target_utility = 0.0;

    /* Fill candidate rows: two-pass (targets first) to fit within cap */
    int included[GO_MM]; memset(included, 0, sizeof(included));
    size_t written = 0;
    for (int pass = 0; pass < 2; pass++) {
        for (int m = 0; m < GO_MM; m++) {
            if (!go_legal(&b, m) || included[m]) continue;
            int tgt = (sc[m] >= best - 0.001f) ? 1 : 0;
            if ((pass == 0 && !tgt) || (pass == 1 && tgt)) continue;
            if (written >= MT_MAX_CANDIDATES) break;

            nn_real *row = &candidate_inputs[written * MT_FEATURE_COUNT];
            mt_fill_base_row(config, row, MT_TASK_GO, written, GO_MM);

            /* Global state features  */
            int bk = 0, wh = 0;
            for (int i = 0; i < GO_CL; i++) {
                if (b.board[i] == 1) bk++; else if (b.board[i] == -1) wh++; }
            row[MT_F_COMMON + 0] = (nn_real)bk / (nn_real)GO_CL;
            row[MT_F_COMMON + 1] = (nn_real)wh / (nn_real)GO_CL;
            row[MT_F_COMMON + 2] = (nn_real)b.moves / (nn_real)GO_MG;
            row[MT_F_COMMON + 3] = b.to_play == 1 ? (nn_real)1 : (nn_real)0;
            row[MT_F_COMMON + 4] = b.passes > 0 ? (nn_real)1 : (nn_real)0;

            /* Move-local features */
            if (m == GO_CL) {
                row[MT_F_STATE + 0] = (nn_real)1;  /* is_pass */
            } else {
                int x = m % GO_BS, y = m / GO_BS;
                row[MT_F_STATE + 1] = (nn_real)x / (nn_real)(GO_BS - 1);
                row[MT_F_STATE + 2] = (nn_real)y / (nn_real)(GO_BS - 1);
                int no = 0, np = 0, ne = 0;
                for (int d = 0; d < 4; d++) { int n = m + GO_DR[d];
                    if ((d == 1 && go_ol(m)) || (d == 2 && go_or(m))) continue;
                    if (!go_ib(n)) continue;
                    if (b.board[n] == b.to_play) no++;
                    else if (b.board[n] == -b.to_play) np++; else ne++; }
                row[MT_F_STATE + 3] = (nn_real)no / 4.0f;
                row[MT_F_STATE + 4] = (nn_real)np / 4.0f;
                row[MT_F_STATE + 5] = (nn_real)ne / 4.0f;
                row[MT_F_STATE + 6] = go_sa(&b, m) ? (nn_real)1 : (nn_real)0;
                row[MT_F_STATE + 7] = go_cap(&b, m) ? (nn_real)1 : (nn_real)0;
                row[MT_F_STATE + 8] = go_save(&b, m) ? (nn_real)1 : (nn_real)0;
                int caps = 0;
                for (int d = 0; d < 4; d++) { int n = m + GO_DR[d];
                    if ((d == 1 && go_ol(m)) || (d == 2 && go_or(m))) continue;
                    if (go_ib(n) && b.board[n] == -b.to_play && go_dead(&b, n)) caps++; }
                row[MT_F_STATE + 9] = (nn_real)caps / 4.0f;
                GoBoard cp = b; cp.board[m] = cp.to_play;
                int libs = go_lib(&cp, m);
                row[MT_F_STATE + 10] = libs > 0 ? 1.0f / (nn_real)libs : (nn_real)0;
            }

            if (tgt && *target_index == (size_t)-1) {
                *target_index = written;
                *target_utility = (double)best;
            }

            included[m] = 1;
            written++;
        }
    }

    return written;
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
    case MT_TASK_GO:
        return mt_fill_go_sample(config,
                                 train ? 8000u + seed_or_index : seed_or_index,
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

static int mt_checked_mul(size_t a, size_t b, size_t *out)
{
    if (out == NULL || (a != 0 && b > SIZE_MAX / a)) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static void *mt_calloc_array(size_t count, size_t size)
{
    size_t bytes = 0;
    if (count == 0 || size == 0 || !mt_checked_mul(count, size, &bytes)) {
        return NULL;
    }
    (void)bytes;
    return calloc(count, size);
}

static nn_real mt_relu(nn_real value)
{
    return value > (nn_real)0 ? value : (nn_real)0;
}

static nn_real mt_relu_grad(nn_real activation)
{
    return activation > (nn_real)0 ? (nn_real)1 : (nn_real)0;
}

static nn_real mt_random_weight(uint32_t *rng, size_t fan_in, size_t fan_out)
{
    double denom = (double)(fan_in + fan_out);
    double scale = denom <= 0.0 ? 0.01 : sqrt(6.0 / denom);
    return (nn_real)((2.0 * mt_rng_unit(rng) - 1.0) * scale);
}

static void mt_init_dense_weights(nn_real *weights,
                                  size_t input_count,
                                  size_t output_count,
                                  uint32_t *rng)
{
    for (size_t output = 0; output < output_count; ++output) {
        for (size_t input = 0; input < input_count; ++input) {
            weights[output * input_count + input] =
                mt_random_weight(rng, input_count, output_count);
        }
    }
}

static MTTaskHeadedScorer *mt_task_heads_create(const MTConfig *config, unsigned int seed)
{
    MTTaskHeadedScorer *model = calloc(1, sizeof(*model));
    if (model == NULL) {
        return NULL;
    }

    model->input_count = MT_FEATURE_COUNT;
    model->hidden1 = config == NULL ? (size_t)64 : config->hidden1;
    model->hidden2 = config == NULL ? (size_t)32 : config->hidden2;
    model->candidate_capacity = MT_MAX_CANDIDATES;

    size_t w1_count = 0;
    size_t w2_count = 0;
    size_t head_count = 0;
    size_t a1_count = 0;
    size_t a2_count = 0;
    if (!mt_checked_mul(model->hidden1, model->input_count, &w1_count) ||
        !mt_checked_mul(model->hidden2, model->hidden1, &w2_count) ||
        !mt_checked_mul((size_t)MT_TASK_COUNT, model->hidden2, &head_count) ||
        !mt_checked_mul(model->candidate_capacity, model->hidden1, &a1_count) ||
        !mt_checked_mul(model->candidate_capacity, model->hidden2, &a2_count)) {
        free(model);
        return NULL;
    }

    model->w1 = mt_calloc_array(w1_count, sizeof(*model->w1));
    model->b1 = mt_calloc_array(model->hidden1, sizeof(*model->b1));
    model->w2 = mt_calloc_array(w2_count, sizeof(*model->w2));
    model->b2 = mt_calloc_array(model->hidden2, sizeof(*model->b2));
    model->head_w = mt_calloc_array(head_count, sizeof(*model->head_w));
    model->head_b = mt_calloc_array((size_t)MT_TASK_COUNT, sizeof(*model->head_b));
    model->a1 = mt_calloc_array(a1_count, sizeof(*model->a1));
    model->a2 = mt_calloc_array(a2_count, sizeof(*model->a2));
    model->scores = mt_calloc_array(model->candidate_capacity, sizeof(*model->scores));
    model->probabilities =
        mt_calloc_array(model->candidate_capacity, sizeof(*model->probabilities));
    model->delta1 = mt_calloc_array(a1_count, sizeof(*model->delta1));
    model->delta2 = mt_calloc_array(a2_count, sizeof(*model->delta2));
    model->grad_w1 = mt_calloc_array(w1_count, sizeof(*model->grad_w1));
    model->grad_b1 = mt_calloc_array(model->hidden1, sizeof(*model->grad_b1));
    model->grad_w2 = mt_calloc_array(w2_count, sizeof(*model->grad_w2));
    model->grad_b2 = mt_calloc_array(model->hidden2, sizeof(*model->grad_b2));
    model->grad_head_w = mt_calloc_array(model->hidden2, sizeof(*model->grad_head_w));

    if (model->w1 == NULL || model->b1 == NULL || model->w2 == NULL ||
        model->b2 == NULL || model->head_w == NULL || model->head_b == NULL ||
        model->a1 == NULL || model->a2 == NULL || model->scores == NULL ||
        model->probabilities == NULL || model->delta1 == NULL ||
        model->delta2 == NULL || model->grad_w1 == NULL ||
        model->grad_b1 == NULL || model->grad_w2 == NULL ||
        model->grad_b2 == NULL || model->grad_head_w == NULL) {
        free(model->w1);
        free(model->b1);
        free(model->w2);
        free(model->b2);
        free(model->head_w);
        free(model->head_b);
        free(model->a1);
        free(model->a2);
        free(model->scores);
        free(model->probabilities);
        free(model->delta1);
        free(model->delta2);
        free(model->grad_w1);
        free(model->grad_b1);
        free(model->grad_w2);
        free(model->grad_b2);
        free(model->grad_head_w);
        free(model);
        return NULL;
    }

    uint32_t rng = seed == 0 ? 1u : seed;
    mt_init_dense_weights(model->w1, model->input_count, model->hidden1, &rng);
    mt_init_dense_weights(model->w2, model->hidden1, model->hidden2, &rng);
    for (size_t task = 0; task < MT_TASK_COUNT; ++task) {
        nn_real *head = &model->head_w[task * model->hidden2];
        for (size_t i = 0; i < model->hidden2; ++i) {
            head[i] = mt_random_weight(&rng, model->hidden2, 1u);
        }
    }

    return model;
}

static void mt_task_heads_free(MTTaskHeadedScorer *model)
{
    if (model == NULL) {
        return;
    }
    free(model->w1);
    free(model->b1);
    free(model->w2);
    free(model->b2);
    free(model->head_w);
    free(model->head_b);
    free(model->a1);
    free(model->a2);
    free(model->scores);
    free(model->probabilities);
    free(model->delta1);
    free(model->delta2);
    free(model->grad_w1);
    free(model->grad_b1);
    free(model->grad_w2);
    free(model->grad_b2);
    free(model->grad_head_w);
    free(model);
}

static void mt_task_heads_forward(MTTaskHeadedScorer *model,
                                  MTTask task,
                                  const nn_real *candidate_inputs,
                                  size_t candidate_count)
{
    const nn_real *head = &model->head_w[(size_t)task * model->hidden2];
    nn_real head_bias = model->head_b[(size_t)task];
    for (size_t sample = 0; sample < candidate_count; ++sample) {
        const nn_real *input = &candidate_inputs[sample * model->input_count];
        nn_real *a1 = &model->a1[sample * model->hidden1];
        nn_real *a2 = &model->a2[sample * model->hidden2];

        for (size_t h = 0; h < model->hidden1; ++h) {
            const nn_real *weights = &model->w1[h * model->input_count];
            nn_real sum = model->b1[h];
            for (size_t i = 0; i < model->input_count; ++i) {
                sum += weights[i] * input[i];
            }
            a1[h] = mt_relu(sum);
        }

        for (size_t h = 0; h < model->hidden2; ++h) {
            const nn_real *weights = &model->w2[h * model->hidden1];
            nn_real sum = model->b2[h];
            for (size_t i = 0; i < model->hidden1; ++i) {
                sum += weights[i] * a1[i];
            }
            a2[h] = mt_relu(sum);
        }

        nn_real score = head_bias;
        for (size_t h = 0; h < model->hidden2; ++h) {
            score += head[h] * a2[h];
        }
        model->scores[sample] = score;
    }
}

static int mt_task_heads_score_action_rows(MTTaskHeadedScorer *model,
                                           MTTask task,
                                           const nn_real *candidate_inputs,
                                           size_t candidate_count,
                                           nn_real *scores,
                                           nn_real *probabilities)
{
    if (model == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        candidate_count > model->candidate_capacity || scores == NULL ||
        (size_t)task >= MT_TASK_COUNT) {
        return 0;
    }
    mt_task_heads_forward(model, task, candidate_inputs, candidate_count);
    memcpy(scores, model->scores, candidate_count * sizeof(*scores));
    if (probabilities != NULL) {
        nn_softmax(scores, probabilities, candidate_count);
    }
    return 1;
}

static void mt_task_heads_zero_gradients(MTTaskHeadedScorer *model)
{
    memset(model->grad_w1, 0, model->hidden1 * model->input_count * sizeof(*model->grad_w1));
    memset(model->grad_b1, 0, model->hidden1 * sizeof(*model->grad_b1));
    memset(model->grad_w2, 0, model->hidden2 * model->hidden1 * sizeof(*model->grad_w2));
    memset(model->grad_b2, 0, model->hidden2 * sizeof(*model->grad_b2));
    memset(model->grad_head_w, 0, model->hidden2 * sizeof(*model->grad_head_w));
}

static void mt_task_heads_apply_gradients(MTTaskHeadedScorer *model,
                                          MTTask task,
                                          nn_real grad_head_b,
                                          nn_real learning_rate)
{
    for (size_t i = 0; i < model->hidden1 * model->input_count; ++i) {
        model->w1[i] -= learning_rate * model->grad_w1[i];
    }
    for (size_t i = 0; i < model->hidden1; ++i) {
        model->b1[i] -= learning_rate * model->grad_b1[i];
    }
    for (size_t i = 0; i < model->hidden2 * model->hidden1; ++i) {
        model->w2[i] -= learning_rate * model->grad_w2[i];
    }
    for (size_t i = 0; i < model->hidden2; ++i) {
        model->b2[i] -= learning_rate * model->grad_b2[i];
    }

    nn_real *head = &model->head_w[(size_t)task * model->hidden2];
    for (size_t i = 0; i < model->hidden2; ++i) {
        head[i] -= learning_rate * model->grad_head_w[i];
    }
    model->head_b[(size_t)task] -= learning_rate * grad_head_b;
}

static nn_real mt_task_heads_train_action_scorer(MTTaskHeadedScorer *model,
                                                 MTTask task,
                                                 const nn_real *candidate_inputs,
                                                 size_t candidate_count,
                                                 size_t target_index,
                                                 nn_real learning_rate)
{
    if (model == NULL || candidate_inputs == NULL || candidate_count == 0 ||
        candidate_count > model->candidate_capacity || target_index >= candidate_count ||
        learning_rate <= (nn_real)0 || (size_t)task >= MT_TASK_COUNT) {
        return (nn_real)NAN;
    }

    mt_task_heads_forward(model, task, candidate_inputs, candidate_count);
    nn_softmax(model->scores, model->probabilities, candidate_count);
    nn_real loss = (nn_real)(-log(mt_safe_probability((double)model->probabilities[target_index])));
    mt_task_heads_zero_gradients(model);

    nn_real *head = &model->head_w[(size_t)task * model->hidden2];
    nn_real grad_head_b = 0;
    for (size_t sample = 0; sample < candidate_count; ++sample) {
        const nn_real *input = &candidate_inputs[sample * model->input_count];
        const nn_real *a1 = &model->a1[sample * model->hidden1];
        const nn_real *a2 = &model->a2[sample * model->hidden2];
        nn_real *delta2 = &model->delta2[sample * model->hidden2];
        nn_real *delta1 = &model->delta1[sample * model->hidden1];
        nn_real dscore = model->probabilities[sample] -
                         (sample == target_index ? (nn_real)1 : (nn_real)0);

        grad_head_b += dscore;
        for (size_t h = 0; h < model->hidden2; ++h) {
            model->grad_head_w[h] += dscore * a2[h];
            delta2[h] = dscore * head[h] * mt_relu_grad(a2[h]);
        }

        for (size_t h = 0; h < model->hidden2; ++h) {
            const nn_real d = delta2[h];
            nn_real *grad_w = &model->grad_w2[h * model->hidden1];
            model->grad_b2[h] += d;
            for (size_t i = 0; i < model->hidden1; ++i) {
                grad_w[i] += d * a1[i];
            }
        }

        for (size_t h = 0; h < model->hidden1; ++h) {
            nn_real sum = 0;
            for (size_t next = 0; next < model->hidden2; ++next) {
                sum += model->w2[next * model->hidden1 + h] * delta2[next];
            }
            delta1[h] = sum * mt_relu_grad(a1[h]);
        }

        for (size_t h = 0; h < model->hidden1; ++h) {
            const nn_real d = delta1[h];
            nn_real *grad_w = &model->grad_w1[h * model->input_count];
            model->grad_b1[h] += d;
            for (size_t i = 0; i < model->input_count; ++i) {
                grad_w[i] += d * input[i];
            }
        }
    }

    mt_task_heads_apply_gradients(model, task, grad_head_b, learning_rate);
    return loss;
}

static size_t mt_task_heads_parameter_count(const MTTaskHeadedScorer *model)
{
    if (model == NULL) {
        return 0;
    }
    size_t total = 0;
    size_t term = 0;
    if (!mt_checked_mul(model->hidden1, model->input_count, &term) ||
        term > SIZE_MAX - total) {
        return 0;
    }
    total += term;
    if (model->hidden1 > SIZE_MAX - total) {
        return 0;
    }
    total += model->hidden1;
    if (!mt_checked_mul(model->hidden2, model->hidden1, &term) ||
        term > SIZE_MAX - total) {
        return 0;
    }
    total += term;
    if (model->hidden2 > SIZE_MAX - total) {
        return 0;
    }
    total += model->hidden2;
    if (!mt_checked_mul((size_t)MT_TASK_COUNT, model->hidden2 + 1u, &term) ||
        term > SIZE_MAX - total) {
        return 0;
    }
    return total + term;
}

static size_t mt_task_heads_forward_flop_count(const MTTaskHeadedScorer *model)
{
    if (model == NULL) {
        return 0;
    }
    size_t total = 0;
    size_t per_h1 = 2u * model->input_count + 1u;
    size_t per_h2 = 2u * model->hidden1 + 1u;
    size_t per_head = 2u * model->hidden2 + 1u;
    if (model->hidden1 > (SIZE_MAX - total) / per_h1) {
        return 0;
    }
    total += model->hidden1 * per_h1;
    if (model->hidden2 > (SIZE_MAX - total) / per_h2) {
        return 0;
    }
    total += model->hidden2 * per_h2;
    if (per_head > SIZE_MAX - total) {
        return 0;
    }
    return total + per_head;
}

static int mt_scorer_score_action_rows(MTScorer *scorer,
                                       MTTask task,
                                       const nn_real *candidate_inputs,
                                       size_t candidate_count,
                                       nn_real *scores,
                                       nn_real *probabilities)
{
    if (scorer == NULL) {
        return 0;
    }
    switch (scorer->kind) {
    case MT_SCORER_NN:
        return nn_score_action_rows(scorer->net,
                                    candidate_inputs,
                                    candidate_count,
                                    scores,
                                    probabilities);
    case MT_SCORER_TASK_HEADS:
        return mt_task_heads_score_action_rows(scorer->task_heads,
                                               task,
                                               candidate_inputs,
                                               candidate_count,
                                               scores,
                                               probabilities);
    }
    return 0;
}

static size_t mt_scorer_parameter_count(const MTScorer *scorer)
{
    if (scorer == NULL) {
        return 0;
    }
    return scorer->kind == MT_SCORER_TASK_HEADS
               ? mt_task_heads_parameter_count(scorer->task_heads)
               : nn_parameter_count(scorer->net);
}

static size_t mt_scorer_forward_flop_count(const MTScorer *scorer)
{
    if (scorer == NULL) {
        return 0;
    }
    return scorer->kind == MT_SCORER_TASK_HEADS
               ? mt_task_heads_forward_flop_count(scorer->task_heads)
               : nn_forward_flop_count(scorer->net);
}

static uint64_t mt_hash_arc_prediction(const uint8_t *prediction, size_t count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < count; ++i) {
        hash ^= (uint64_t)prediction[i] + UINT64_C(0x9e3779b97f4a7c15) +
                (uint64_t)i;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
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

static void mt_print_epoch_json(const MTConfig *config,
                                const char *model_kind,
                                MTTask task,
                                size_t epoch,
                                size_t epoch_updates,
                                size_t train_updates,
                                double loss_sum)
{
    char task_set[128];
    char hidden[32];
    FILE *out = mt_json_output(config);
    mt_task_set_name(config->task_mask, task_set, sizeof(task_set));
    mt_hidden_text(config, hidden, sizeof(hidden));
    double denom = epoch_updates == 0 ? 1.0 : (double)epoch_updates;
    fprintf(out,
            "{\"schema\":\"crlp.multitask.epoch.v1\","
            "\"report_version\":1,"
            "\"task\":\"multi_domain\","
            "\"domain\":\"%s\","
            "\"model_kind\":\"%s\","
            "\"training_order\":\"%s\","
            "\"feature_envelope\":\"%s\","
            "\"task_set\":\"%s\","
            "\"epoch\":%zu,"
            "\"epochs\":%zu,"
            "\"steps_per_epoch\":%zu,"
            "\"epoch_updates\":%zu,"
            "\"train_updates\":%zu,"
            "\"loss_mean\":%.8f,"
            "\"hidden\":\"%s\","
            "\"learning_rate\":%.8f,"
            "\"seed\":%llu",
            mt_task_name(task),
            model_kind == NULL ? "shared" : model_kind,
            mt_training_order_name((MTTrainingOrder)config->training_order),
            mt_feature_envelope_name(config),
            task_set,
            epoch,
            config->epochs,
            config->steps_per_epoch,
            epoch_updates,
            train_updates,
            loss_sum / denom,
            hidden,
            (double)config->learning_rate,
            (unsigned long long)config->seed);
    mt_print_report_metadata_json(out, config);
    fputs("}\n", out);
}

static int mt_train_mixed_one_epoch(NeuralNet *net,
                                    const MTData *data,
                                    const MTConfig *config,
                                    size_t epoch_index,
                                    uint32_t *rng,
                                    size_t updates[MT_TASK_COUNT],
                                    nn_real *candidate_inputs)
{
    MTTask active_tasks[MT_TASK_COUNT];
    size_t active_count;
    double losses[MT_TASK_COUNT] = {0.0, 0.0, 0.0, 0.0};
    size_t epoch_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
    if (net == NULL || data == NULL || config == NULL || rng == NULL ||
        updates == NULL || candidate_inputs == NULL) {
        return 0;
    }
    active_count = mt_fill_active_tasks(config->task_mask,
                                        active_tasks,
                                        MT_TASK_COUNT);
    if (active_count == 0U) {
        return 0;
    }
    for (size_t step = 0; step < config->steps_per_epoch; ++step) {
        size_t global_step = epoch_index * config->steps_per_epoch + step;
        MTTask task = mt_choose_training_task(config,
                                              active_tasks,
                                              active_count,
                                              global_step,
                                              rng);
        uint64_t sample = (uint64_t)mt_rng_next(rng);
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
            return 0;
        }
        nn_real loss = nn_train_action_scorer(net,
                                              candidate_inputs,
                                              candidate_count,
                                              target_index,
                                              config->learning_rate);
        if (isnan((double)loss)) {
            return 0;
        }
        losses[task] += (double)loss;
        ++epoch_updates[task];
        ++updates[task];
    }

    if (!config->quiet) {
        printf("multitask epoch %zu", epoch_index + 1U);
        for (size_t i = 0; i < active_count; ++i) {
            size_t task = (size_t)active_tasks[i];
            double denom = epoch_updates[task] == 0 ? 1.0 : (double)epoch_updates[task];
            printf(" %s_loss=%.4f", mt_task_name((MTTask)task), losses[task] / denom);
        }
        putchar('\n');
    }
    for (size_t i = 0; i < active_count; ++i) {
        size_t task = (size_t)active_tasks[i];
        mt_print_epoch_json(config,
                            "shared",
                            (MTTask)task,
                            epoch_index + 1U,
                            epoch_updates[task],
                            updates[task],
                            losses[task]);
    }
    return 1;
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
        if (!mt_train_mixed_one_epoch(net,
                                      data,
                                      config,
                                      epoch,
                                      &rng,
                                      updates,
                                      candidate_inputs)) {
            free(candidate_inputs);
            return 0;
        }
    }

    free(candidate_inputs);
    return 1;
}

static int mt_train_mixed_task_heads_one_epoch(MTTaskHeadedScorer *model,
                                               const MTData *data,
                                               const MTConfig *config,
                                               size_t epoch_index,
                                               uint32_t *rng,
                                               size_t updates[MT_TASK_COUNT],
                                               nn_real *candidate_inputs)
{
    MTTask active_tasks[MT_TASK_COUNT];
    size_t active_count;
    double losses[MT_TASK_COUNT] = {0.0, 0.0, 0.0, 0.0};
    size_t epoch_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
    if (model == NULL || data == NULL || config == NULL || rng == NULL ||
        updates == NULL || candidate_inputs == NULL) {
        return 0;
    }
    active_count = mt_fill_active_tasks(config->task_mask,
                                        active_tasks,
                                        MT_TASK_COUNT);
    if (active_count == 0U) {
        return 0;
    }
    for (size_t step = 0; step < config->steps_per_epoch; ++step) {
        size_t global_step = epoch_index * config->steps_per_epoch + step;
        MTTask task = mt_choose_training_task(config,
                                              active_tasks,
                                              active_count,
                                              global_step,
                                              rng);
        uint64_t sample = (uint64_t)mt_rng_next(rng);
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
            return 0;
        }
        nn_real loss = mt_task_heads_train_action_scorer(model,
                                                         task,
                                                         candidate_inputs,
                                                         candidate_count,
                                                         target_index,
                                                         config->learning_rate);
        if (isnan((double)loss)) {
            return 0;
        }
        losses[task] += (double)loss;
        ++epoch_updates[task];
        ++updates[task];
    }

    if (!config->quiet) {
        printf("task_heads epoch %zu", epoch_index + 1U);
        for (size_t i = 0; i < active_count; ++i) {
            size_t task = (size_t)active_tasks[i];
            double denom = epoch_updates[task] == 0 ? 1.0 : (double)epoch_updates[task];
            printf(" %s_loss=%.4f", mt_task_name((MTTask)task), losses[task] / denom);
        }
        putchar('\n');
    }
    for (size_t i = 0; i < active_count; ++i) {
        size_t task = (size_t)active_tasks[i];
        mt_print_epoch_json(config,
                            "shared_task_heads",
                            (MTTask)task,
                            epoch_index + 1U,
                            epoch_updates[task],
                            updates[task],
                            losses[task]);
    }
    return 1;
}

static int mt_train_mixed_task_heads(MTTaskHeadedScorer *model,
                                     const MTData *data,
                                     const MTConfig *config,
                                     size_t updates[MT_TASK_COUNT])
{
    uint32_t rng =
        (uint32_t)(config->seed ^ (config->seed >> 32) ^ UINT64_C(0x27182818));
    MTTask active_tasks[MT_TASK_COUNT];
    size_t active_count = mt_fill_active_tasks(config->task_mask,
                                               active_tasks,
                                               MT_TASK_COUNT);
    nn_real *candidate_inputs =
        calloc((size_t)MT_MAX_CANDIDATES * MT_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (model == NULL || active_count == 0 || candidate_inputs == NULL) {
        free(candidate_inputs);
        return 0;
    }

    memset(updates, 0, MT_TASK_COUNT * sizeof(*updates));
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        if (!mt_train_mixed_task_heads_one_epoch(model,
                                                 data,
                                                 config,
                                                 epoch,
                                                 &rng,
                                                 updates,
                                                 candidate_inputs)) {
            free(candidate_inputs);
            return 0;
        }
    }

    free(candidate_inputs);
    return 1;
}

static int mt_train_single_task_steps(NeuralNet *net,
                                      const MTData *data,
                                      const MTConfig *config,
                                      MTTask task,
                                      uint32_t *rng,
                                      nn_real *candidate_inputs,
                                      size_t updates,
                                      double *loss_sum_out)
{
    double loss_sum = 0.0;
    if (net == NULL || data == NULL || config == NULL || rng == NULL ||
        candidate_inputs == NULL) {
        return 0;
    }
    for (size_t step = 0; step < updates; ++step) {
        uint64_t sample = (uint64_t)mt_rng_next(rng);
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
            return 0;
        }
        nn_real loss = nn_train_action_scorer(net,
                                              candidate_inputs,
                                              candidate_count,
                                              target_index,
                                              config->learning_rate);
        if (isnan((double)loss)) {
            return 0;
        }
        loss_sum += (double)loss;
    }
    if (loss_sum_out != NULL) {
        *loss_sum_out = loss_sum;
    }
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
    double loss_sum = 0.0;
    if (candidate_inputs == NULL) {
        return 0;
    }

    if (!mt_train_single_task_steps(net,
                                    data,
                                    config,
                                    task,
                                    &rng,
                                    candidate_inputs,
                                    updates,
                                    &loss_sum)) {
        free(candidate_inputs);
        return 0;
    }

    if (!config->quiet) {
        printf("single_task %s updates=%zu loss=%.4f\n",
               mt_task_name(task),
               updates,
               loss_sum / (updates == 0 ? 1.0 : (double)updates));
    }
    free(candidate_inputs);
    return 1;
}

static MTMetrics mt_evaluate_task(MTScorer *scorer,
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
            !mt_scorer_score_action_rows(scorer,
                                         task,
                                         candidate_inputs,
                                         candidate_count,
                                         scores,
                                         probabilities) ||
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

static int mt_compute_arc_fixed_diagnostics(MTScorer *scorer,
                                            const MTConfig *config,
                                            MTArcFixedDiagnostics *diag)
{
    if (scorer == NULL || config == NULL || diag == NULL) {
        return 0;
    }
    memset(diag, 0, sizeof(*diag));
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
        return 0;
    }

    for (size_t cell = 0; cell < MT_ARC_OUTPUT_CELLS; ++cell) {
        size_t target_index = 0;
        double ignored_utility = 0.0;
        size_t candidate_count =
            mt_fill_arc_cell_rows(config,
                                  MT_ARC_0520FDE7_TEST_INPUT,
                                  MT_ARC_0520FDE7_TEST_OUTPUT,
                                  cell,
                                  candidate_inputs,
                                  &target_index,
                                  &ignored_utility);
        if (candidate_count == 0 || target_index >= candidate_count ||
            !mt_scorer_score_action_rows(scorer,
                                         MT_TASK_ARC_AGI,
                                         candidate_inputs,
                                         candidate_count,
                                         scores,
                                         probabilities) ||
            !nn_rank_action_scores(scores, candidate_count, ranked)) {
            continue;
        }
        uint8_t predicted = (uint8_t)ranked[0];
        uint8_t target = (uint8_t)target_index;
        diag->target[cell] = target;
        diag->predicted[cell] = predicted;
        diag->confidence[cell] = (double)probabilities[ranked[0]];
        diag->target_probability[cell] = (double)probabilities[target_index];
        diag->nll += -log(mt_safe_probability((double)probabilities[target_index]));
        ++diag->cells;
        if (target < MT_ARC_ACTION_COUNT) {
            ++diag->target_color_count[target];
        }
        if (predicted < MT_ARC_ACTION_COUNT) {
            ++diag->predicted_color_count[predicted];
        }
        if (predicted == target) {
            ++diag->correct_cells;
            if (target < MT_ARC_ACTION_COUNT) {
                ++diag->correct_target_color_count[target];
            }
        }
    }
    if (diag->cells != 0) {
        diag->nll /= (double)diag->cells;
    }
    diag->grid_exact = diag->cells == MT_ARC_OUTPUT_CELLS &&
                       diag->correct_cells == MT_ARC_OUTPUT_CELLS;
    diag->prediction_hash = mt_hash_arc_prediction(diag->predicted, MT_ARC_OUTPUT_CELLS);

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free(ranked);
    return diag->cells == MT_ARC_OUTPUT_CELLS;
}

static void mt_print_u8_json_array(FILE *out,
                                   const char *name,
                                   const uint8_t *values,
                                   size_t count)
{
    fprintf(out, ",\"%s\":[", name);
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            fputc(',', out);
        }
        fprintf(out, "%u", (unsigned int)values[i]);
    }
    fputc(']', out);
}

static void mt_print_size_json_array(FILE *out,
                                     const char *name,
                                     const size_t *values,
                                     size_t count)
{
    fprintf(out, ",\"%s\":[", name);
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            fputc(',', out);
        }
        fprintf(out, "%zu", values[i]);
    }
    fputc(']', out);
}

static void mt_print_double_json_array(FILE *out,
                                       const char *name,
                                       const double *values,
                                       size_t count)
{
    fprintf(out, ",\"%s\":[", name);
    for (size_t i = 0; i < count; ++i) {
        if (i != 0) {
            fputc(',', out);
        }
        fprintf(out, "%.8f", values[i]);
    }
    fputc(']', out);
}

static void mt_print_arc_fixed_diagnostics_json(MTScorer *scorer,
                                                MTTask task,
                                                const MTConfig *config,
                                                FILE *out)
{
    if (task != MT_TASK_ARC_AGI || config == NULL ||
        config->arc_eval_suite != MT_ARC_EVAL_0520FDE7) {
        return;
    }
    MTArcFixedDiagnostics diag;
    if (!mt_compute_arc_fixed_diagnostics(scorer, config, &diag)) {
        return;
    }
    double cells = diag.cells == 0 ? 1.0 : (double)diag.cells;
    fprintf(out,
            ",\"arc_fixed_grid_exact\":%s,"
            "\"arc_fixed_cell_accuracy\":%.8f,"
            "\"arc_fixed_correct_cells\":%zu,"
            "\"arc_fixed_cells\":%zu,"
            "\"arc_fixed_nll\":%.8f,"
            "\"arc_fixed_prediction_hash\":\"%016llx\"",
            diag.grid_exact ? "true" : "false",
            (double)diag.correct_cells / cells,
            diag.correct_cells,
            diag.cells,
            diag.nll,
            (unsigned long long)diag.prediction_hash);
    mt_print_u8_json_array(out, "arc_fixed_targets", diag.target, MT_ARC_OUTPUT_CELLS);
    mt_print_u8_json_array(out, "arc_fixed_predictions", diag.predicted, MT_ARC_OUTPUT_CELLS);
    mt_print_double_json_array(out,
                               "arc_fixed_confidence",
                               diag.confidence,
                               MT_ARC_OUTPUT_CELLS);
    mt_print_double_json_array(out,
                               "arc_fixed_target_probability",
                               diag.target_probability,
                               MT_ARC_OUTPUT_CELLS);
    mt_print_size_json_array(out,
                             "arc_fixed_target_color_count",
                             diag.target_color_count,
                             MT_ARC_ACTION_COUNT);
    mt_print_size_json_array(out,
                             "arc_fixed_predicted_color_count",
                             diag.predicted_color_count,
                             MT_ARC_ACTION_COUNT);
    mt_print_size_json_array(out,
                             "arc_fixed_correct_target_color_count",
                             diag.correct_target_color_count,
                             MT_ARC_ACTION_COUNT);
}

static void mt_print_reliability_bins_json(FILE *out, const MTMetrics *metrics)
{
    int first = 1;
    fputs(",\"reliability_bins\":[", out);
    if (metrics != NULL) {
        for (size_t bin = 0; bin < MT_CALIBRATION_BINS; ++bin) {
            if (metrics->bin_count[bin] == 0) {
                continue;
            }
            if (!first) {
                fputc(',', out);
            }
            first = 0;
            fprintf(out,
                    "{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
                    "\"confidence\":%.8f,\"accuracy\":%.8f}",
                    (double)bin / (double)MT_CALIBRATION_BINS,
                    (double)(bin + 1u) / (double)MT_CALIBRATION_BINS,
                    metrics->bin_count[bin],
                    metrics->bin_confidence[bin],
                    metrics->bin_accuracy[bin]);
        }
    }
    fputc(']', out);
}

static void mt_print_arc_eval_json(FILE *out, const MTConfig *config, MTTask task)
{
    if (task != MT_TASK_ARC_AGI || config == NULL) {
        return;
    }
    fprintf(out, ",\"arc_eval_suite\":\"%s\"", mt_arc_eval_suite_name(config->arc_eval_suite));
    if (config->arc_eval_suite == MT_ARC_EVAL_OOD_SAME_ENVELOPE) {
        if (config->arc_ood_task >= 0 &&
            config->arc_ood_task < (int)MT_ARC_OOD_TASK_COUNT) {
            const MTArcOODTask *ood = &MT_ARC_OOD_TASKS[config->arc_ood_task];
            fprintf(out,
                    ",\"arc_eval_task\":\"%s\","
                    "\"arc_eval_family\":\"%s\","
                    "\"arc_eval_task_index\":%d,"
                    "\"arc_eval_task_count\":1",
                    ood->name,
                    ood->family_name,
                    config->arc_ood_task);
        } else {
            fprintf(out,
                    ",\"arc_eval_task\":\"all\","
                    "\"arc_eval_family\":\"mixed\","
                    "\"arc_eval_task_count\":%d",
                    MT_ARC_OOD_TASK_COUNT);
        }
    } else {
        fputs(",\"arc_eval_task\":\"0520fde7\","
              "\"arc_eval_family\":\"intersection\","
              "\"arc_eval_task_count\":1",
              out);
    }
}

static int mt_report_rows_add(MTReportRows *rows,
                              const MTConfig *config,
                              const char *model_kind,
                              MTTask task,
                              size_t train_updates,
                              size_t parameters,
                              size_t forward_flops,
                              const MTMetrics *metrics)
{
    double samples;
    MTReportRow *next;
    if (rows == NULL || config == NULL || model_kind == NULL || metrics == NULL) {
        return 1;
    }
    if (rows->count == rows->capacity) {
        size_t next_capacity = rows->capacity == 0U ? 64U : rows->capacity * 2U;
        MTReportRow *next_items =
            (MTReportRow *)realloc(rows->items, next_capacity * sizeof(*rows->items));
        if (next_items == NULL) {
            return 0;
        }
        rows->items = next_items;
        rows->capacity = next_capacity;
    }
    samples = metrics->samples == 0 ? 1.0 : (double)metrics->samples;
    next = &rows->items[rows->count++];
    next->eval_suite = config->adapter_eval_suite != NULL &&
                               config->adapter_eval_suite[0] != '\0'
                           ? config->adapter_eval_suite
                           : (config->arc_eval_suite == MT_ARC_EVAL_OOD_SAME_ENVELOPE ? "ood"
                                                                                       : "fixed");
    next->domain = mt_task_name(task);
    next->model_kind = model_kind;
    mt_hidden_text(config, next->hidden, sizeof(next->hidden));
    next->seed = config->seed;
    next->epochs = config->epochs;
    next->top1 = (double)metrics->top1 / samples;
    next->nll = metrics->nll;
    next->ece = metrics->ece;
    next->decision_utility = metrics->utility;
    next->train_updates = (double)train_updates;
    next->parameters = (double)parameters;
    next->forward_flops = (double)forward_flops;
    return 1;
}

static void mt_print_metrics_json(MTScorer *scorer,
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
    FILE *out = mt_json_output(config);
    size_t parameters = mt_scorer_parameter_count(scorer);
    size_t forward_flops = mt_scorer_forward_flop_count(scorer);
    mt_task_set_name(config->task_mask, task_set, sizeof(task_set));
    mt_task_set_name(config->eval_task_mask, eval_task_set, sizeof(eval_task_set));
    mt_hidden_text(config, hidden, sizeof(hidden));
    fprintf(out,
            "{\"schema\":\"crlp.multitask.summary.v1\","
            "\"task\":\"multi_domain\","
            "\"domain\":\"%s\"",
            mt_task_name(task));
    mt_print_arc_eval_json(out, config, task);
    fprintf(out,
            ","
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
            task_set,
            eval_task_set,
            model_kind,
            training_order,
            mt_feature_envelope_name(config),
            train_updates,
            config->epochs,
            config->steps_per_epoch,
            hidden,
            parameters,
            forward_flops,
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
    mt_print_report_metadata_json(out, config);
    mt_print_arc_fixed_diagnostics_json(scorer, task, config, out);
    mt_print_reliability_bins_json(out, &metrics);
    fputs("}\n", out);
    (void)mt_report_rows_add(config->report_rows,
                             config,
                             model_kind,
                             task,
                             train_updates,
                             parameters,
                             forward_flops,
                             &metrics);
}

static int mt_report_rows_append(MTReportRows *rows, const MTReportRow *row)
{
    if (rows == NULL || row == NULL) {
        return 0;
    }
    if (rows->count == rows->capacity) {
        size_t next_capacity = rows->capacity == 0U ? 64U : rows->capacity * 2U;
        MTReportRow *next_items =
            (MTReportRow *)realloc(rows->items, next_capacity * sizeof(*rows->items));
        if (next_items == NULL) {
            return 0;
        }
        rows->items = next_items;
        rows->capacity = next_capacity;
    }
    rows->items[rows->count++] = *row;
    return 1;
}

static const char *mt_json_field_value(const char *line, const char *key)
{
    char pattern[96];
    int written;
    const char *match;
    if (line == NULL || key == NULL || key[0] == '\0') {
        return NULL;
    }
    written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written < 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    match = strstr(line, pattern);
    return match == NULL ? NULL : match + (size_t)written;
}

static const char *mt_skip_json_space(const char *cursor)
{
    while (cursor != NULL &&
           (*cursor == ' ' || *cursor == '\t' || *cursor == '\r' || *cursor == '\n')) {
        ++cursor;
    }
    return cursor;
}

static int mt_json_line_get_string(const char *line,
                                   const char *key,
                                   char *out,
                                   size_t out_size)
{
    const char *cursor = mt_skip_json_space(mt_json_field_value(line, key));
    size_t used = 0U;
    if (cursor == NULL || out == NULL || out_size == 0U || *cursor != '"') {
        return 0;
    }
    ++cursor;
    while (*cursor != '\0' && *cursor != '"') {
        char ch = *cursor++;
        if (ch == '\\') {
            ch = *cursor++;
            if (ch == '\0') {
                return 0;
            }
            switch (ch) {
            case '"':
            case '\\':
            case '/':
                break;
            case 'b':
                ch = '\b';
                break;
            case 'f':
                ch = '\f';
                break;
            case 'n':
                ch = '\n';
                break;
            case 'r':
                ch = '\r';
                break;
            case 't':
                ch = '\t';
                break;
            default:
                return 0;
            }
        }
        if (used + 1U >= out_size) {
            return 0;
        }
        out[used++] = ch;
    }
    if (*cursor != '"') {
        return 0;
    }
    out[used] = '\0';
    return 1;
}

static int mt_json_line_get_double(const char *line, const char *key, double *out)
{
    const char *cursor = mt_skip_json_space(mt_json_field_value(line, key));
    char *end = NULL;
    double value;
    if (cursor == NULL || out == NULL) {
        return 0;
    }
    errno = 0;
    value = strtod(cursor, &end);
    if (end == cursor || errno == ERANGE || !isfinite(value)) {
        return 0;
    }
    *out = value;
    return 1;
}

static int mt_json_line_get_u64(const char *line, const char *key, uint64_t *out)
{
    const char *cursor = mt_skip_json_space(mt_json_field_value(line, key));
    char *end = NULL;
    unsigned long long value;
    if (cursor == NULL || out == NULL || *cursor == '-' || *cursor == '+') {
        return 0;
    }
    errno = 0;
    value = strtoull(cursor, &end, 10);
    if (end == cursor || errno == ERANGE) {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static const char *mt_report_domain_canonical(const char *domain)
{
    for (int task = 0; task < (int)MT_TASK_COUNT; ++task) {
        const char *name = mt_task_name((MTTask)task);
        if (strcmp(domain, name) == 0) {
            return name;
        }
    }
    return NULL;
}

static const char *mt_report_model_canonical(const char *model_kind)
{
    if (strcmp(model_kind, "shared") == 0) {
        return "shared";
    }
    if (strcmp(model_kind, "shared_task_heads") == 0) {
        return "shared_task_heads";
    }
    if (strcmp(model_kind, "single_task") == 0) {
        return "single_task";
    }
    return NULL;
}

static const char *mt_report_eval_suite_canonical(const char *eval_suite)
{
    if (strcmp(eval_suite, "fixed") == 0) {
        return "fixed";
    }
    if (strcmp(eval_suite, "ood") == 0) {
        return "ood";
    }
    return NULL;
}

static const char *mt_report_eval_suite_from_line(const char *line)
{
    char value[32];
    const char *suite;
    if (mt_json_line_get_string(line, "adapter_eval_suite", value, sizeof(value)) ||
        mt_json_line_get_string(line, "eval_suite", value, sizeof(value))) {
        suite = mt_report_eval_suite_canonical(value);
        return suite == NULL ? NULL : suite;
    }
    if (mt_json_line_get_string(line, "arc_eval_task", value, sizeof(value)) &&
        strcmp(value, "all") == 0) {
        return "ood";
    }
    return "fixed";
}

static int mt_parse_report_row_json(const char *line, MTReportRow *row)
{
    char domain[64];
    char model_kind[64];
    char hidden[32];
    uint64_t seed = 0U;
    uint64_t epochs = 0U;
    const char *canonical_domain;
    const char *canonical_model;
    const char *eval_suite;
    MTReportRow parsed;
    if (line == NULL || row == NULL) {
        return -1;
    }
    if (strstr(line, "\"schema\":\"crlp.multitask.summary.v1\"") == NULL) {
        return 0;
    }
    if (!mt_json_line_get_string(line, "domain", domain, sizeof(domain)) ||
        !mt_json_line_get_string(line, "model_kind", model_kind, sizeof(model_kind)) ||
        !mt_json_line_get_string(line, "hidden", hidden, sizeof(hidden)) ||
        !mt_json_line_get_u64(line, "seed", &seed) ||
        !mt_json_line_get_u64(line, "epochs", &epochs)) {
        return -1;
    }
    canonical_domain = mt_report_domain_canonical(domain);
    canonical_model = mt_report_model_canonical(model_kind);
    eval_suite = mt_report_eval_suite_from_line(line);
    if (canonical_domain == NULL || canonical_model == NULL || eval_suite == NULL) {
        return -1;
    }
    memset(&parsed, 0, sizeof(parsed));
    parsed.eval_suite = eval_suite;
    parsed.domain = canonical_domain;
    parsed.model_kind = canonical_model;
    snprintf(parsed.hidden, sizeof(parsed.hidden), "%s", hidden);
    parsed.seed = seed;
    parsed.epochs = (size_t)epochs;
    if (!mt_json_line_get_double(line, "top1", &parsed.top1) ||
        !mt_json_line_get_double(line, "nll", &parsed.nll) ||
        !mt_json_line_get_double(line, "ece", &parsed.ece) ||
        !mt_json_line_get_double(line, "decision_utility", &parsed.decision_utility) ||
        !mt_json_line_get_double(line, "train_updates", &parsed.train_updates) ||
        !mt_json_line_get_double(line, "parameters", &parsed.parameters) ||
        !mt_json_line_get_double(line, "forward_flops", &parsed.forward_flops)) {
        return -1;
    }
    *row = parsed;
    return 1;
}

static int mt_report_rows_load_jsonl(MTReportRows *rows, const char *path)
{
    char line[65536];
    FILE *fp;
    if (rows == NULL || path == NULL) {
        return -1;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        MTReportRow row;
        int parsed;
        if (strchr(line, '\n') == NULL && !feof(fp)) {
            fclose(fp);
            return -1;
        }
        parsed = mt_parse_report_row_json(line, &row);
        if (parsed < 0) {
            fclose(fp);
            return -1;
        }
        if (parsed > 0 && !mt_report_rows_append(rows, &row)) {
            fclose(fp);
            return -1;
        }
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
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

static void mt_print_arc_ood_task_breakdown(MTScorer *scorer,
                                            const MTData *data,
                                            const MTConfig *config,
                                            const char *model_kind,
                                            const char *training_order,
                                            size_t train_updates)
{
    if (scorer == NULL || data == NULL || config == NULL ||
        config->arc_eval_suite != MT_ARC_EVAL_OOD_SAME_ENVELOPE ||
        config->arc_ood_task >= 0 || !config->report_arc_ood_tasks ||
        (config->eval_task_mask & mt_task_bit(MT_TASK_ARC_AGI)) == 0) {
        return;
    }
    for (size_t task_index = 0; task_index < MT_ARC_OOD_TASK_COUNT; ++task_index) {
        MTConfig task_config = *config;
        task_config.arc_ood_task = (int)task_index;
        MTMetrics metrics =
            mt_evaluate_task(scorer, data, &task_config, MT_TASK_ARC_AGI, config->eval_samples);
        mt_print_metrics_json(scorer,
                              model_kind,
                              training_order,
                              MT_TASK_ARC_AGI,
                              train_updates,
                              &task_config,
                              metrics);
    }
}

static void mt_print_usage(FILE *stream)
{
    fputs("usage: multitask_domain [--epochs count] [--steps-per-epoch count] "
          "[--eval-samples count] [--learning-rate rate] [--seed seed] "
          "[--hidden h1xh2] "
          "[--training-order uniform_random|round_robin|arc_heavy|prime_light] "
          "[--arc-eval-suite 0520fde7|ood] [--arc-ood-task all|index|name] "
          "[--arc-ood-task-breakdown] "
          "[--tasks prime,grid,claim,arc|all] [--eval-tasks prime,grid,claim,arc|all] "
          "[--task-heads] [--adapter-heads-report --output-dir path --seeds csv] "
          "[--tiny-net-plateau-audit --capacities csv --epoch-grid csv "
          "--resume --shard-index n --shard-count n] "
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
    config->arc_eval_suite = MT_ARC_EVAL_0520FDE7;
    config->arc_ood_task = -1;
    config->report_arc_ood_tasks = 0;
    config->run_task_head_model = 0;
    config->adapter_heads_report = 0;
    config->tiny_net_plateau_audit = 0;
    config->tiny_net_resume = 0;
    config->tiny_net_shard_index = 1U;
    config->tiny_net_shard_count = 1U;
    config->quiet = 0;
    config->output_dir = "results/2026-05-17-multitask-roadmap";
    config->report_seeds = "20240516,20240517,20240518";
    config->tiny_net_capacities = "8x4,16x8,32x16,64x32";
    config->tiny_net_epoch_grid = "1,2,4,8,16,32";
    config->roadmap_item = NULL;
    config->adapter_eval_suite = NULL;
    config->transfer_regression_margin = 0.005;
    config->plateau_threshold = 0.005;
    config->plateau_window = 3U;
    config->json_output = NULL;
    config->report_rows = NULL;

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
        } else if (strcmp(argv[i], "--arc-eval-suite") == 0 && i + 1 < argc) {
            int suite = MT_ARC_EVAL_0520FDE7;
            if (!mt_parse_arc_eval_suite(argv[++i], &suite)) {
                return 0;
            }
            config->arc_eval_suite = suite;
        } else if (strcmp(argv[i], "--arc-ood-task") == 0 && i + 1 < argc) {
            if (!mt_parse_arc_ood_task(argv[++i], &config->arc_ood_task)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-ood-task-breakdown") == 0) {
            config->report_arc_ood_tasks = 1;
        } else if (strcmp(argv[i], "--task-heads") == 0 ||
                   strcmp(argv[i], "--shared-task-heads") == 0) {
            config->run_task_head_model = 1;
        } else if (strcmp(argv[i], "--adapter-heads-report") == 0) {
            config->adapter_heads_report = 1;
            config->run_task_head_model = 1;
            config->quiet = 1;
            config->epochs = 16U;
            config->steps_per_epoch = 12000U;
            config->task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                                mt_task_bit(MT_TASK_GRID_PATH) |
                                mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                                mt_task_bit(MT_TASK_ARC_AGI);
            config->eval_task_mask = config->task_mask;
        } else if (strcmp(argv[i], "--tiny-net-plateau-audit") == 0) {
            config->tiny_net_plateau_audit = 1;
            config->run_task_head_model = 1;
            config->quiet = 1;
            config->task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                                mt_task_bit(MT_TASK_GRID_PATH) |
                                mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                                mt_task_bit(MT_TASK_ARC_AGI);
            config->eval_task_mask = config->task_mask;
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            config->output_dir = argv[++i];
        } else if (strcmp(argv[i], "--seeds") == 0 && i + 1 < argc) {
            config->report_seeds = argv[++i];
        } else if (strcmp(argv[i], "--capacities") == 0 && i + 1 < argc) {
            config->tiny_net_capacities = argv[++i];
        } else if (strcmp(argv[i], "--epoch-grid") == 0 && i + 1 < argc) {
            config->tiny_net_epoch_grid = argv[++i];
        } else if (strcmp(argv[i], "--transfer-regression-margin") == 0 &&
                   i + 1 < argc) {
            char *end = NULL;
            double parsed = strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0') {
                return 0;
            }
            config->transfer_regression_margin = parsed;
        } else if (strcmp(argv[i], "--plateau-threshold") == 0 && i + 1 < argc) {
            char *end = NULL;
            double parsed = strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0') {
                return 0;
            }
            config->plateau_threshold = parsed;
        } else if (strcmp(argv[i], "--plateau-window") == 0 && i + 1 < argc) {
            if (!mt_parse_size_text(argv[++i], &config->plateau_window)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--resume") == 0 ||
                   strcmp(argv[i], "--tiny-net-resume") == 0) {
            config->tiny_net_resume = 1;
        } else if (strcmp(argv[i], "--shard") == 0 && i + 1 < argc) {
            if (!mt_parse_shard_text(argv[++i],
                                     &config->tiny_net_shard_index,
                                     &config->tiny_net_shard_count)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--shard-index") == 0 && i + 1 < argc) {
            if (!mt_parse_size_text(argv[++i], &config->tiny_net_shard_index)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--shard-count") == 0 && i + 1 < argc) {
            if (!mt_parse_size_text(argv[++i], &config->tiny_net_shard_count)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--no-task-heads") == 0) {
            config->run_task_head_model = 0;
        } else if (strcmp(argv[i], "--quiet") == 0) {
            config->quiet = 1;
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
    if (config->tiny_net_shard_count == 0U ||
        config->tiny_net_shard_index == 0U ||
        config->tiny_net_shard_index > config->tiny_net_shard_count) {
        return 0;
    }
    return 1;
}

static int mt_run_once(const MTConfig *config)
{
    MTData data;
    if (!mt_prepare_data(&data)) {
        fputs("failed to prepare multitask datasets\n", stderr);
        return EXIT_FAILURE;
    }

    NeuralNet *shared = mt_create_model(config, (unsigned int)(config->seed & 0xffffffffu));
    if (shared == NULL) {
        mt_free_data(&data);
        return EXIT_FAILURE;
    }

    char task_set[128];
    char eval_task_set[128];
    char hidden[32];
    mt_task_set_name(config->task_mask, task_set, sizeof(task_set));
    mt_task_set_name(config->eval_task_mask, eval_task_set, sizeof(eval_task_set));
    mt_hidden_text(config, hidden, sizeof(hidden));
    if (!config->quiet) {
        printf("[multitask_domain] feature_envelope=%s "
               "task_set=%s eval_task_set=%s epochs=%zu steps_per_epoch=%zu "
               "eval_samples=%zu hidden=%s training_order=%s arc_eval_suite=%s "
               "task_heads=%s lr=%.6f seed=%llu\n",
               mt_feature_envelope_name(config),
               task_set,
               eval_task_set,
               config->epochs,
               config->steps_per_epoch,
               config->eval_samples,
               hidden,
               mt_training_order_name((MTTrainingOrder)config->training_order),
               mt_arc_eval_suite_name(config->arc_eval_suite),
               config->run_task_head_model ? "on" : "off",
               (double)config->learning_rate,
               (unsigned long long)config->seed);
    }

    size_t shared_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
    if (!mt_train_mixed(shared, &data, config, shared_updates)) {
        nn_free(shared);
        mt_free_data(&data);
        return EXIT_FAILURE;
    }
    MTScorer shared_scorer;
    shared_scorer.kind = MT_SCORER_NN;
    shared_scorer.net = shared;
    shared_scorer.task_heads = NULL;

    MTTask active_tasks[MT_TASK_COUNT];
    size_t active_count = mt_fill_active_tasks(config->task_mask,
                                               active_tasks,
                                               MT_TASK_COUNT);
    MTTask eval_tasks[MT_TASK_COUNT];
    size_t eval_count = mt_fill_active_tasks(config->eval_task_mask,
                                             eval_tasks,
                                             MT_TASK_COUNT);
    for (size_t task_index = 0; task_index < eval_count; ++task_index) {
        MTTask task = eval_tasks[task_index];
        MTMetrics metrics =
            mt_evaluate_task(&shared_scorer, &data, config, task, config->eval_samples);
        mt_print_metrics_json(&shared_scorer,
                              "shared",
                              mt_training_order_name((MTTrainingOrder)config->training_order),
                              task,
                              shared_updates[task],
                              config,
                              metrics);
        if (!config->quiet) {
            mt_print_metrics_line("shared", task, metrics);
        }
        if (task == MT_TASK_ARC_AGI) {
            mt_print_arc_ood_task_breakdown(&shared_scorer,
                                            &data,
                                            config,
                                            "shared",
                                            mt_training_order_name(
                                                (MTTrainingOrder)config->training_order),
                                            shared_updates[task]);
        }
    }

    if (config->run_task_head_model) {
        MTTaskHeadedScorer *task_heads =
            mt_task_heads_create(config,
                                 (unsigned int)((config->seed + UINT64_C(303)) &
                                                UINT64_C(0xffffffff)));
        size_t task_head_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
        if (task_heads == NULL ||
            !mt_train_mixed_task_heads(task_heads, &data, config, task_head_updates)) {
            mt_task_heads_free(task_heads);
            nn_free(shared);
            mt_free_data(&data);
            return EXIT_FAILURE;
        }
        MTScorer task_head_scorer;
        task_head_scorer.kind = MT_SCORER_TASK_HEADS;
        task_head_scorer.net = NULL;
        task_head_scorer.task_heads = task_heads;
        for (size_t task_index = 0; task_index < eval_count; ++task_index) {
            MTTask task = eval_tasks[task_index];
            MTMetrics metrics =
                mt_evaluate_task(&task_head_scorer, &data, config, task, config->eval_samples);
            mt_print_metrics_json(&task_head_scorer,
                                  "shared_task_heads",
                                  mt_training_order_name((MTTrainingOrder)config->training_order),
                                  task,
                                  task_head_updates[task],
                                  config,
                                  metrics);
            if (!config->quiet) {
                mt_print_metrics_line("shared_task_heads", task, metrics);
            }
            if (task == MT_TASK_ARC_AGI) {
                mt_print_arc_ood_task_breakdown(&task_head_scorer,
                                                &data,
                                                config,
                                                "shared_task_heads",
                                                mt_training_order_name(
                                                    (MTTrainingOrder)config->training_order),
                                                task_head_updates[task]);
            }
        }
        mt_task_heads_free(task_heads);
    }

    if (config->run_task_baselines) {
        size_t baseline_updates =
            (config->epochs * config->steps_per_epoch) / active_count;
        if (baseline_updates == 0) {
            baseline_updates = 1;
        }
        for (size_t task_index = 0; task_index < active_count; ++task_index) {
            MTTask task = active_tasks[task_index];
            if ((config->eval_task_mask & mt_task_bit(task)) == 0) {
                continue;
            }
            NeuralNet *single =
                mt_create_model(config,
                                (unsigned int)((config->seed + 101u + task_index) &
                                               0xffffffffu));
            if (single == NULL ||
                !mt_train_single_task(single, &data, config, task, baseline_updates)) {
                nn_free(single);
                nn_free(shared);
                mt_free_data(&data);
                return EXIT_FAILURE;
            }
            MTScorer single_scorer;
            single_scorer.kind = MT_SCORER_NN;
            single_scorer.net = single;
            single_scorer.task_heads = NULL;
            MTMetrics metrics =
                mt_evaluate_task(&single_scorer, &data, config, task, config->eval_samples);
            mt_print_metrics_json(&single_scorer,
                                  "single_task",
                                  "domain_only_random",
                                  task,
                                  baseline_updates,
                                  config,
                                  metrics);
            if (!config->quiet) {
                mt_print_metrics_line("single_task", task, metrics);
            }
            if (task == MT_TASK_ARC_AGI) {
                mt_print_arc_ood_task_breakdown(&single_scorer,
                                                &data,
                                                config,
                                                "single_task",
                                                "domain_only_random",
                                                baseline_updates);
            }
            nn_free(single);
        }
    }

    nn_free(shared);
    mt_free_data(&data);
    return EXIT_SUCCESS;
}

typedef struct {
    const char *eval_suite;
    const char *domain;
    const char *model_kind;
    int present;
    long seeds;
    long rows;
    double top1_mean;
    double top1_stdev;
    double nll_mean;
    double nll_stdev;
    double ece_mean;
    double ece_stdev;
    double decision_utility_mean;
    double train_updates_mean;
    double parameters_mean;
    double forward_flops_mean;
} MTReportSummary;

typedef struct {
    const char *eval_suite;
    const char *domain;
    int present;
    long seeds;
    double headed_top1_mean;
    double shared_top1_mean;
    double single_top1_mean;
    double top1_delta_headed_minus_shared;
    double top1_delta_headed_minus_single;
    double headed_nll_mean;
    double shared_nll_mean;
    double single_nll_mean;
    double nll_delta_headed_minus_shared;
    double nll_delta_headed_minus_single;
    double headed_ece_mean;
    double shared_ece_mean;
    double single_ece_mean;
    double ece_delta_headed_minus_shared;
    double ece_delta_headed_minus_single;
    double headed_parameters_mean;
    double shared_parameters_mean;
    double single_parameters_mean;
    double headed_forward_flops_mean;
    double shared_forward_flops_mean;
    double single_forward_flops_mean;
    double headed_train_updates_mean;
    double shared_train_updates_mean;
    double single_train_updates_mean;
} MTReportDelta;

static double mt_nan(void)
{
    return NAN;
}

static int mt_mkdir_p(const char *path)
{
    char buffer[4096];
    size_t len;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    len = strlen(path);
    if (len >= sizeof(buffer)) {
        return -1;
    }
    memcpy(buffer, path, len + 1U);
    for (size_t i = 1U; i < len; ++i) {
        if (buffer[i] == '/') {
            buffer[i] = '\0';
            if (buffer[0] != '\0' && mkdir(buffer, 0755) != 0 && errno != EEXIST) {
                return -1;
            }
            buffer[i] = '/';
        }
    }
    if (mkdir(buffer, 0755) != 0 && errno != EEXIST) {
        return -1;
    }
    return 0;
}

static int mt_join_path(char *out, size_t out_size, const char *dir, const char *name)
{
    int written;
    if (out == NULL || out_size == 0U || dir == NULL || name == NULL) {
        return -1;
    }
    written = snprintf(out,
                       out_size,
                       "%s%s%s",
                       dir,
                       dir[0] != '\0' && dir[strlen(dir) - 1U] == '/' ? "" : "/",
                       name);
    return written < 0 || (size_t)written >= out_size ? -1 : 0;
}

static int mt_file_has_content(const char *path)
{
    struct stat st;
    return path != NULL && stat(path, &st) == 0 && st.st_size > 0;
}

static int mt_copy_file_to(FILE *out, const char *path)
{
    char buffer[16384];
    FILE *fp;
    if (out == NULL || path == NULL) {
        return -1;
    }
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    for (;;) {
        size_t n = fread(buffer, 1U, sizeof(buffer), fp);
        if (n > 0U && fwrite(buffer, 1U, n, out) != n) {
            fclose(fp);
            return -1;
        }
        if (n < sizeof(buffer)) {
            if (ferror(fp)) {
                fclose(fp);
                return -1;
            }
            break;
        }
    }
    fclose(fp);
    return 0;
}

static void mt_today_iso(char out[11])
{
    time_t now = time(NULL);
    struct tm tm_value;
#if defined(_POSIX_VERSION)
    if (localtime_r(&now, &tm_value) == NULL) {
        strcpy(out, "1970-01-01");
        return;
    }
#else
    struct tm *tmp = localtime(&now);
    if (tmp == NULL) {
        strcpy(out, "1970-01-01");
        return;
    }
    tm_value = *tmp;
#endif
    if (strftime(out, 11U, "%Y-%m-%d", &tm_value) == 0U) {
        strcpy(out, "1970-01-01");
    }
}

static int mt_parse_seed_list(const char *text, uint64_t *seeds, size_t capacity, size_t *count)
{
    const char *cursor = text == NULL || text[0] == '\0' ? "20240516,20240517,20240518" : text;
    size_t used = 0U;
    while (*cursor != '\0') {
        char *end = NULL;
        unsigned long long parsed;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        errno = 0;
        parsed = strtoull(cursor, &end, 10);
        if (end == cursor || errno != 0 || used == capacity) {
            return 0;
        }
        seeds[used++] = (uint64_t)parsed;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ':') {
            char *range_end = NULL;
            unsigned long long last;
            ++cursor;
            errno = 0;
            last = strtoull(cursor, &range_end, 10);
            if (range_end == cursor || errno != 0 || last < parsed) {
                return 0;
            }
            cursor = range_end;
            while (parsed < last) {
                if (used == capacity) {
                    return 0;
                }
                ++parsed;
                seeds[used++] = (uint64_t)parsed;
            }
        }
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0') {
            return 0;
        }
    }
    *count = used;
    return used > 0U;
}

static int mt_seed_seen(uint64_t *seeds, size_t count, uint64_t seed)
{
    for (size_t i = 0U; i < count; ++i) {
        if (seeds[i] == seed) {
            return 1;
        }
    }
    return 0;
}

static double mt_report_stdev(double sum, double sum_sq, long rows)
{
    if (rows < 2L) {
        return 0.0;
    }
    double mean = sum / (double)rows;
    double variance = (sum_sq - (double)rows * mean * mean) / (double)(rows - 1L);
    return variance <= 0.0 ? 0.0 : sqrt(variance);
}

static int mt_summary_collect(const MTReportRows *rows,
                              const char *eval_suite,
                              const char *domain,
                              const char *model_kind,
                              MTReportSummary *summary)
{
    uint64_t seeds[128];
    size_t seed_count = 0U;
    double top1_sum = 0.0, top1_sq = 0.0;
    double nll_sum = 0.0, nll_sq = 0.0;
    double ece_sum = 0.0, ece_sq = 0.0;
    double utility_sum = 0.0;
    double updates_sum = 0.0;
    double params_sum = 0.0;
    double flops_sum = 0.0;
    long count = 0L;
    memset(summary, 0, sizeof(*summary));
    summary->eval_suite = eval_suite;
    summary->domain = domain;
    summary->model_kind = model_kind;
    for (size_t i = 0U; rows != NULL && i < rows->count; ++i) {
        const MTReportRow *row = &rows->items[i];
        if (strcmp(row->eval_suite, eval_suite) != 0 ||
            strcmp(row->domain, domain) != 0 ||
            strcmp(row->model_kind, model_kind) != 0) {
            continue;
        }
        ++count;
        top1_sum += row->top1;
        top1_sq += row->top1 * row->top1;
        nll_sum += row->nll;
        nll_sq += row->nll * row->nll;
        ece_sum += row->ece;
        ece_sq += row->ece * row->ece;
        utility_sum += row->decision_utility;
        updates_sum += row->train_updates;
        params_sum += row->parameters;
        flops_sum += row->forward_flops;
        if (!mt_seed_seen(seeds, seed_count, row->seed) && seed_count < sizeof(seeds) / sizeof(seeds[0])) {
            seeds[seed_count++] = row->seed;
        }
    }
    if (count == 0L) {
        return 0;
    }
    summary->present = 1;
    summary->rows = count;
    summary->seeds = (long)seed_count;
    summary->top1_mean = top1_sum / (double)count;
    summary->top1_stdev = mt_report_stdev(top1_sum, top1_sq, count);
    summary->nll_mean = nll_sum / (double)count;
    summary->nll_stdev = mt_report_stdev(nll_sum, nll_sq, count);
    summary->ece_mean = ece_sum / (double)count;
    summary->ece_stdev = mt_report_stdev(ece_sum, ece_sq, count);
    summary->decision_utility_mean = utility_sum / (double)count;
    summary->train_updates_mean = updates_sum / (double)count;
    summary->parameters_mean = params_sum / (double)count;
    summary->forward_flops_mean = flops_sum / (double)count;
    return 1;
}

static const MTReportSummary *mt_find_summary(const MTReportSummary *summaries,
                                              size_t count,
                                              const char *eval_suite,
                                              const char *domain,
                                              const char *model_kind)
{
    for (size_t i = 0U; i < count; ++i) {
        if (summaries[i].present &&
            strcmp(summaries[i].eval_suite, eval_suite) == 0 &&
            strcmp(summaries[i].domain, domain) == 0 &&
            strcmp(summaries[i].model_kind, model_kind) == 0) {
            return &summaries[i];
        }
    }
    return NULL;
}

static double mt_summary_value(const MTReportSummary *summary, double value)
{
    return summary == NULL ? mt_nan() : value;
}

static MTReportDelta mt_delta_build(const MTReportSummary *summaries,
                                    size_t summary_count,
                                    const char *eval_suite,
                                    const char *domain)
{
    const MTReportSummary *headed =
        mt_find_summary(summaries, summary_count, eval_suite, domain, "shared_task_heads");
    const MTReportSummary *shared =
        mt_find_summary(summaries, summary_count, eval_suite, domain, "shared");
    const MTReportSummary *single =
        mt_find_summary(summaries, summary_count, eval_suite, domain, "single_task");
    MTReportDelta delta;
    memset(&delta, 0, sizeof(delta));
    delta.eval_suite = eval_suite;
    delta.domain = domain;
    if (headed == NULL) {
        return delta;
    }
    delta.present = 1;
    delta.seeds = headed->seeds;
    delta.headed_top1_mean = headed->top1_mean;
    delta.shared_top1_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->top1_mean);
    delta.single_top1_mean = mt_summary_value(single, single == NULL ? 0.0 : single->top1_mean);
    delta.top1_delta_headed_minus_shared =
        shared == NULL ? mt_nan() : headed->top1_mean - shared->top1_mean;
    delta.top1_delta_headed_minus_single =
        single == NULL ? mt_nan() : headed->top1_mean - single->top1_mean;
    delta.headed_nll_mean = headed->nll_mean;
    delta.shared_nll_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->nll_mean);
    delta.single_nll_mean = mt_summary_value(single, single == NULL ? 0.0 : single->nll_mean);
    delta.nll_delta_headed_minus_shared =
        shared == NULL ? mt_nan() : headed->nll_mean - shared->nll_mean;
    delta.nll_delta_headed_minus_single =
        single == NULL ? mt_nan() : headed->nll_mean - single->nll_mean;
    delta.headed_ece_mean = headed->ece_mean;
    delta.shared_ece_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->ece_mean);
    delta.single_ece_mean = mt_summary_value(single, single == NULL ? 0.0 : single->ece_mean);
    delta.ece_delta_headed_minus_shared =
        shared == NULL ? mt_nan() : headed->ece_mean - shared->ece_mean;
    delta.ece_delta_headed_minus_single =
        single == NULL ? mt_nan() : headed->ece_mean - single->ece_mean;
    delta.headed_parameters_mean = headed->parameters_mean;
    delta.shared_parameters_mean = mt_summary_value(shared, shared == NULL ? 0.0 : shared->parameters_mean);
    delta.single_parameters_mean = mt_summary_value(single, single == NULL ? 0.0 : single->parameters_mean);
    delta.headed_forward_flops_mean = headed->forward_flops_mean;
    delta.shared_forward_flops_mean =
        mt_summary_value(shared, shared == NULL ? 0.0 : shared->forward_flops_mean);
    delta.single_forward_flops_mean =
        mt_summary_value(single, single == NULL ? 0.0 : single->forward_flops_mean);
    delta.headed_train_updates_mean = headed->train_updates_mean;
    delta.shared_train_updates_mean =
        mt_summary_value(shared, shared == NULL ? 0.0 : shared->train_updates_mean);
    delta.single_train_updates_mean =
        mt_summary_value(single, single == NULL ? 0.0 : single->train_updates_mean);
    return delta;
}

static const MTReportDelta *mt_find_delta(const MTReportDelta *deltas,
                                          size_t count,
                                          const char *eval_suite,
                                          const char *domain)
{
    for (size_t i = 0U; i < count; ++i) {
        if (deltas[i].present &&
            strcmp(deltas[i].eval_suite, eval_suite) == 0 &&
            strcmp(deltas[i].domain, domain) == 0) {
            return &deltas[i];
        }
    }
    return NULL;
}

static void mt_csv_double(FILE *fp, double value)
{
    if (!isnan(value)) {
        fprintf(fp, "%.17g", value);
    }
}

static void mt_json_number_or_null(FILE *fp, double value)
{
    if (isnan(value)) {
        fputs("null", fp);
    } else {
        fprintf(fp, "%.17g", value);
    }
}

static void mt_format_pct(char *out, size_t out_size, double value)
{
    if (isnan(value)) {
        snprintf(out, out_size, "-");
    } else {
        snprintf(out, out_size, "%.2f%%", 100.0 * value);
    }
}

static void mt_format_delta(char *out, size_t out_size, double value)
{
    if (isnan(value)) {
        snprintf(out, out_size, "-");
    } else {
        snprintf(out, out_size, "%+.2fpp", 100.0 * value);
    }
}

static void mt_format_float(char *out, size_t out_size, double value)
{
    if (isnan(value)) {
        snprintf(out, out_size, "-");
    } else {
        snprintf(out, out_size, "%.4f", value);
    }
}

static int mt_write_summary_csv(const char *path,
                                const MTReportSummary *summaries,
                                size_t count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("eval_suite,domain,model_kind,seeds,rows,top1_mean,top1_stdev,nll_mean,nll_stdev,ece_mean,ece_stdev,decision_utility_mean,train_updates_mean,parameters_mean,forward_flops_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTReportSummary *row = &summaries[i];
        if (!row->present) {
            continue;
        }
        fprintf(fp, "%s,%s,%s,%ld,%ld,", row->eval_suite, row->domain, row->model_kind, row->seeds, row->rows);
        mt_csv_double(fp, row->top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->top1_stdev); fputc(',', fp);
        mt_csv_double(fp, row->nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->nll_stdev); fputc(',', fp);
        mt_csv_double(fp, row->ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->ece_stdev); fputc(',', fp);
        mt_csv_double(fp, row->decision_utility_mean); fputc(',', fp);
        mt_csv_double(fp, row->train_updates_mean); fputc(',', fp);
        mt_csv_double(fp, row->parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->forward_flops_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_deltas_csv(const char *path,
                               const MTReportDelta *deltas,
                               size_t count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("eval_suite,domain,seeds,headed_top1_mean,shared_top1_mean,single_top1_mean,top1_delta_headed_minus_shared,top1_delta_headed_minus_single,headed_nll_mean,shared_nll_mean,single_nll_mean,nll_delta_headed_minus_shared,nll_delta_headed_minus_single,headed_ece_mean,shared_ece_mean,single_ece_mean,ece_delta_headed_minus_shared,ece_delta_headed_minus_single,headed_parameters_mean,shared_parameters_mean,single_parameters_mean,headed_forward_flops_mean,shared_forward_flops_mean,single_forward_flops_mean,headed_train_updates_mean,shared_train_updates_mean,single_train_updates_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTReportDelta *row = &deltas[i];
        if (!row->present) {
            continue;
        }
        fprintf(fp, "%s,%s,%ld,", row->eval_suite, row->domain, row->seeds);
        mt_csv_double(fp, row->headed_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->top1_delta_headed_minus_shared); fputc(',', fp);
        mt_csv_double(fp, row->top1_delta_headed_minus_single); fputc(',', fp);
        mt_csv_double(fp, row->headed_nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->nll_delta_headed_minus_shared); fputc(',', fp);
        mt_csv_double(fp, row->nll_delta_headed_minus_single); fputc(',', fp);
        mt_csv_double(fp, row->headed_ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->ece_delta_headed_minus_shared); fputc(',', fp);
        mt_csv_double(fp, row->ece_delta_headed_minus_single); fputc(',', fp);
        mt_csv_double(fp, row->headed_parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->headed_forward_flops_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_forward_flops_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_forward_flops_mean); fputc(',', fp);
        mt_csv_double(fp, row->headed_train_updates_mean); fputc(',', fp);
        mt_csv_double(fp, row->shared_train_updates_mean); fputc(',', fp);
        mt_csv_double(fp, row->single_train_updates_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_decision_jsonl(const char *path,
                                   const MTReportDelta *deltas,
                                   size_t delta_count,
                                   double margin)
{
    const MTReportDelta *fixed_arc = mt_find_delta(deltas, delta_count, "fixed", "arc_agi");
    const MTReportDelta *ood_arc = mt_find_delta(deltas, delta_count, "ood", "arc_agi");
    const MTReportDelta *delayed = mt_find_delta(deltas, delta_count, "fixed", "delayed_claim");
    const MTReportDelta *grid = mt_find_delta(deltas, delta_count, "fixed", "grid_path");
    int complete = fixed_arc != NULL && ood_arc != NULL && delayed != NULL && grid != NULL;
    long seeds = 0L;
    double fixed_arc_gain = fixed_arc == NULL ? mt_nan() : fixed_arc->top1_delta_headed_minus_shared;
    double delayed_delta = delayed == NULL ? mt_nan() : delayed->top1_delta_headed_minus_shared;
    double grid_delta = grid == NULL ? mt_nan() : grid->top1_delta_headed_minus_shared;
    double ood_top1 = ood_arc == NULL ? mt_nan() : ood_arc->top1_delta_headed_minus_single;
    double ood_nll = ood_arc == NULL ? mt_nan() : ood_arc->nll_delta_headed_minus_single;
    int fixed_gate = !isnan(fixed_arc_gain) && fixed_arc_gain > 0.0;
    int delayed_gate = !isnan(delayed_delta) && delayed_delta >= -margin;
    int grid_gate = !isnan(grid_delta) && grid_delta >= -margin;
    int ood_gate = !isnan(ood_top1) && !isnan(ood_nll) && ood_top1 >= 0.0 && ood_nll <= 0.0;
    int promote;
    char date[11];
    FILE *fp;
    complete = complete && !isnan(fixed_arc_gain) && !isnan(delayed_delta) &&
               !isnan(grid_delta) && !isnan(ood_top1) && !isnan(ood_nll);
    if (complete) {
        seeds = fixed_arc->seeds;
        if (ood_arc->seeds < seeds) seeds = ood_arc->seeds;
        if (delayed->seeds < seeds) seeds = delayed->seeds;
        if (grid->seeds < seeds) seeds = grid->seeds;
    }
    promote = complete && seeds >= 3L && fixed_gate && delayed_gate && grid_gate && ood_gate;
    mt_today_iso(date);
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("{\"complete_required_rows\": ", fp);
    fputs(complete ? "true" : "false", fp);
    fprintf(fp, ", \"date\": \"%s\", \"decision\": \"%s\", ", date, promote ? "promote_shared_trunk_heads" : "diagnostic_only");
    fputs("\"delayed_claim_top1_delta_heads_minus_flat_shared\": ", fp); mt_json_number_or_null(fp, delayed_delta);
    fputs(", \"delayed_claim_transfer_preserved\": ", fp); fputs(delayed_gate ? "true" : "false", fp);
    fputs(", \"fixed_arc_improves_vs_flat_shared\": ", fp); fputs(fixed_gate ? "true" : "false", fp);
    fputs(", \"fixed_arc_top1_delta_heads_minus_flat_shared\": ", fp); mt_json_number_or_null(fp, fixed_arc_gain);
    fputs(", \"grid_path_not_materially_regressed\": ", fp); fputs(grid_gate ? "true" : "false", fp);
    fputs(", \"grid_path_top1_delta_heads_minus_flat_shared\": ", fp); mt_json_number_or_null(fp, grid_delta);
    fputs(", \"ood_arc_nll_delta_heads_minus_single_task\": ", fp); mt_json_number_or_null(fp, ood_nll);
    fputs(", \"ood_arc_preserved_vs_single_task\": ", fp); fputs(ood_gate ? "true" : "false", fp);
    fputs(", \"ood_arc_top1_delta_heads_minus_single_task\": ", fp); mt_json_number_or_null(fp, ood_top1);
    fputs(", \"promote_shared_trunk_heads\": ", fp); fputs(promote ? "true" : "false", fp);
    fprintf(fp,
            ", \"report_version\": 1, \"schema\": \"crlp.multitask.adapter_heads_decision.v1\", \"seeds\": %ld, \"transfer_regression_margin\": %.17g}\n",
            seeds,
            margin);
    fclose(fp);
    return 0;
}

static void mt_markdown_top1_row(FILE *fp,
                                 const char *label,
                                 const MTReportSummary *shared,
                                 const MTReportSummary *headed,
                                 const MTReportSummary *single,
                                 const MTReportDelta *delta)
{
    char shared_pct[32], headed_pct[32], single_pct[32], delta_shared[32], delta_single[32];
    mt_format_pct(shared_pct, sizeof(shared_pct), shared == NULL ? mt_nan() : shared->top1_mean);
    mt_format_pct(headed_pct, sizeof(headed_pct), headed == NULL ? mt_nan() : headed->top1_mean);
    mt_format_pct(single_pct, sizeof(single_pct), single == NULL ? mt_nan() : single->top1_mean);
    mt_format_delta(delta_shared,
                    sizeof(delta_shared),
                    delta == NULL ? mt_nan() : delta->top1_delta_headed_minus_shared);
    mt_format_delta(delta_single,
                    sizeof(delta_single),
                    delta == NULL ? mt_nan() : delta->top1_delta_headed_minus_single);
    fprintf(fp,
            "| `%s` | %s | %s | %s | %s | %s |\n",
            label,
            shared_pct,
            headed_pct,
            single_pct,
            delta_shared,
            delta_single);
}

static int mt_write_adapter_report_md(const char *path,
                                      const uint64_t *seeds,
                                      size_t seed_count,
                                      const MTConfig *base,
                                      const MTReportSummary *summaries,
                                      size_t summary_count,
                                      const MTReportDelta *deltas,
                                      size_t delta_count)
{
    FILE *fp = fopen(path, "w");
    char seed_text[512];
    size_t offset = 0U;
    const char *fixed_domains[] = {"prime_gap", "grid_path", "delayed_claim", "arc_agi"};
    const char *models[] = {"shared", "shared_task_heads", "single_task"};
    const MTReportDelta *decision_delta = mt_find_delta(deltas, delta_count, "fixed", "arc_agi");
    (void)decision_delta;
    if (fp == NULL) {
        return -1;
    }
    seed_text[0] = '\0';
    for (size_t i = 0U; i < seed_count; ++i) {
        int written = snprintf(seed_text + offset,
                               sizeof(seed_text) - offset,
                               "%s%llu",
                               i == 0U ? "" : ", ",
                               (unsigned long long)seeds[i]);
        if (written < 0 || (size_t)written >= sizeof(seed_text) - offset) {
            break;
        }
        offset += (size_t)written;
    }
    fprintf(fp,
            "# Multitask Adapter-Headed Sharing\n\n"
            "Seeds: `%s`\n"
            "Budget: `%zu` epochs x `%zu` steps = `%zu` total shared updates.\n"
            "Evaluation samples: `%zu` per domain row.\n\n"
            "This run compares the flat shared scalar scorer, a shared trunk with per-domain scalar heads, and matched single-task baselines under the same 96-feature envelope.\n\n"
            "## Fixed-Suite Top-1\n\n"
            "| Domain | Shared | Task heads | Single | Heads - shared | Heads - single |\n"
            "|---|---:|---:|---:|---:|---:|\n",
            seed_text,
            base->epochs,
            base->steps_per_epoch,
            base->epochs * base->steps_per_epoch,
            base->eval_samples);
    for (size_t i = 0U; i < sizeof(fixed_domains) / sizeof(fixed_domains[0]); ++i) {
        const char *domain = fixed_domains[i];
        mt_markdown_top1_row(fp,
                             domain,
                             mt_find_summary(summaries, summary_count, "fixed", domain, "shared"),
                             mt_find_summary(summaries, summary_count, "fixed", domain, "shared_task_heads"),
                             mt_find_summary(summaries, summary_count, "fixed", domain, "single_task"),
                             mt_find_delta(deltas, delta_count, "fixed", domain));
    }
    fputs("\n## ARC OOD Top-1\n\n"
          "| Suite | Shared | Task heads | Single | Heads - shared | Heads - single |\n"
          "|---|---:|---:|---:|---:|---:|\n",
          fp);
    mt_markdown_top1_row(fp,
                         "arc_ood",
                         mt_find_summary(summaries, summary_count, "ood", "arc_agi", "shared"),
                         mt_find_summary(summaries, summary_count, "ood", "arc_agi", "shared_task_heads"),
                         mt_find_summary(summaries, summary_count, "ood", "arc_agi", "single_task"),
                         mt_find_delta(deltas, delta_count, "ood", "arc_agi"));
    fputs("\n## ARC And Transfer Guardrail\n\n"
          "| Suite | Domain | Model | Top-1 | NLL | ECE | Updates | Params | FLOPs |\n"
          "|---|---|---|---:|---:|---:|---:|---:|---:|\n",
          fp);
    for (size_t pair = 0U; pair < 4U; ++pair) {
        const char *suite = pair == 1U ? "ood" : "fixed";
        const char *domain = pair == 2U ? "delayed_claim" : pair == 3U ? "grid_path" : "arc_agi";
        for (size_t model_index = 0U; model_index < sizeof(models) / sizeof(models[0]); ++model_index) {
            const MTReportSummary *row =
                mt_find_summary(summaries, summary_count, suite, domain, models[model_index]);
            char top1[32], nll[32], ece[32];
            if (row == NULL) {
                continue;
            }
            mt_format_pct(top1, sizeof(top1), row->top1_mean);
            mt_format_float(nll, sizeof(nll), row->nll_mean);
            mt_format_float(ece, sizeof(ece), row->ece_mean);
            fprintf(fp,
                    "| `%s` | `%s` | `%s` | %s | %s | %s | %.0f | %.0f | %.0f |\n",
                    suite,
                    domain,
                    row->model_kind,
                    top1,
                    nll,
                    ece,
                    row->train_updates_mean,
                    row->parameters_mean,
                    row->forward_flops_mean);
        }
    }
    fputs("\n## Decision Rule\n\n"
          "- Decision: see `rmt4_adapter_heads_decision.jsonl`.\n"
          "- Promote this architecture only if task heads improve fixed ARC versus the flat shared scorer, preserve delayed-claim transfer, avoid a material grid regression, and preserve ARC OOD versus the single-task baseline.\n"
          "- If task heads help fixed ARC but hurt OOD or transfer, keep them as diagnostic evidence rather than the default architecture.\n",
          fp);
    fclose(fp);
    return 0;
}

typedef struct {
    size_t hidden1;
    size_t hidden2;
    char label[32];
} MTTinyCapacity;

typedef struct {
    char hidden[32];
    const char *domain;
    const char *model_kind;
    size_t epochs;
    int present;
    long seeds;
    long rows;
    double top1_mean;
    double top1_stdev;
    double top1_min;
    double top1_max;
    double nll_mean;
    double nll_stdev;
    double ece_mean;
    double ece_stdev;
    double parameters_mean;
    double forward_flops_mean;
} MTTinyCurve;

typedef struct {
    char hidden[32];
    const char *domain;
    const char *model_kind;
    int present;
    size_t best_epochs;
    double best_top1_mean;
    double best_top1_stdev;
    size_t last_epochs;
    double last_top1_mean;
    double last_gain;
    double last_window_gain;
    double best_gain_from_first;
    double plateau_threshold;
    size_t plateau_window;
    int best_epoch_in_last_window;
    int plateau_detected;
    double parameters_mean;
    double forward_flops_mean;
} MTTinyPlateau;

typedef struct {
    const char *domain;
    const char *model_kind;
    int present;
    char smallest_hidden[32];
    char largest_hidden[32];
    char best_hidden[32];
    double best_top1_mean;
    double largest_minus_smallest_top1;
    int larger_width_helped;
} MTTinyWidth;

typedef struct {
    const char *domain;
    const char *model_kind;
    int present;
    long capacity_count;
    int capacity_coverage_complete;
    long restart_count;
    long epoch_checkpoint_count;
    long plateau_count;
    int all_plateau_detected;
    int width_verdict_present;
    char best_hidden[32];
    double best_top1_mean;
    int larger_width_helped;
    int audit_ready;
} MTTinyVerdict;

static int mt_parse_capacity_list(const char *text,
                                  MTTinyCapacity *capacities,
                                  size_t capacity,
                                  size_t *count)
{
    const char *cursor = text == NULL || text[0] == '\0' ? "8x4,16x8,32x16,64x32" : text;
    size_t used = 0U;
    while (*cursor != '\0') {
        char token[64];
        size_t len = 0U;
        size_t hidden1 = 0U, hidden2 = 0U;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        while (cursor[len] != '\0' && cursor[len] != ',') {
            ++len;
        }
        while (len > 0U && (cursor[len - 1U] == ' ' || cursor[len - 1U] == '\t')) {
            --len;
        }
        if (len == 0U || len >= sizeof(token) || used == capacity) {
            return 0;
        }
        memcpy(token, cursor, len);
        token[len] = '\0';
        if (!mt_parse_hidden_text(token, &hidden1, &hidden2)) {
            return 0;
        }
        capacities[used].hidden1 = hidden1;
        capacities[used].hidden2 = hidden2;
        snprintf(capacities[used].label, sizeof(capacities[used].label), "%zux%zu", hidden1, hidden2);
        ++used;
        cursor += len;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0') {
            return 0;
        }
    }
    *count = used;
    return used > 0U;
}

static int mt_parse_size_list(const char *text, size_t *values, size_t capacity, size_t *count)
{
    const char *cursor = text == NULL || text[0] == '\0' ? "1,2,4,8,16,32" : text;
    size_t used = 0U;
    while (*cursor != '\0') {
        char *end = NULL;
        unsigned long long parsed;
        while (*cursor == ' ' || *cursor == '\t' || *cursor == ',') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        errno = 0;
        parsed = strtoull(cursor, &end, 10);
        if (end == cursor || errno != 0 || parsed == 0ULL || used == capacity) {
            return 0;
        }
        values[used++] = (size_t)parsed;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == ':') {
            char *range_end = NULL;
            unsigned long long last;
            ++cursor;
            errno = 0;
            last = strtoull(cursor, &range_end, 10);
            if (range_end == cursor || errno != 0 || last < parsed) {
                return 0;
            }
            cursor = range_end;
            while (parsed < last) {
                if (used == capacity) {
                    return 0;
                }
                ++parsed;
                values[used++] = (size_t)parsed;
            }
        }
        if (*cursor == ',') {
            ++cursor;
        } else if (*cursor != '\0') {
            return 0;
        }
    }
    *count = used;
    return used > 0U;
}

static int mt_tiny_curve_matches(const MTTinyCurve *curve,
                                 const MTReportRow *row)
{
    return strcmp(curve->hidden, row->hidden) == 0 &&
           strcmp(curve->domain, row->domain) == 0 &&
           strcmp(curve->model_kind, row->model_kind) == 0 &&
           curve->epochs == row->epochs;
}

static int mt_tiny_collect_curves(const MTReportRows *rows,
                                  MTTinyCurve **curves_out,
                                  size_t *curve_count_out)
{
    MTTinyCurve *curves;
    size_t curve_count = 0U;
    if (rows == NULL || curves_out == NULL || curve_count_out == NULL) {
        return -1;
    }
    curves = (MTTinyCurve *)calloc(rows->count == 0U ? 1U : rows->count, sizeof(*curves));
    if (curves == NULL) {
        return -1;
    }
    for (size_t row_index = 0U; row_index < rows->count; ++row_index) {
        const MTReportRow *row = &rows->items[row_index];
        int found = 0;
        for (size_t i = 0U; i < curve_count; ++i) {
            if (mt_tiny_curve_matches(&curves[i], row)) {
                found = 1;
                break;
            }
        }
        if (!found) {
            MTTinyCurve *curve = &curves[curve_count++];
            curve->present = 1;
            snprintf(curve->hidden, sizeof(curve->hidden), "%s", row->hidden);
            curve->domain = row->domain;
            curve->model_kind = row->model_kind;
            curve->epochs = row->epochs;
        }
    }
    for (size_t i = 0U; i < curve_count; ++i) {
        MTTinyCurve *curve = &curves[i];
        double top1_sum = 0.0, top1_sq = 0.0;
        double nll_sum = 0.0, nll_sq = 0.0;
        double ece_sum = 0.0, ece_sq = 0.0;
        double params_sum = 0.0, flops_sum = 0.0;
        uint64_t seeds[128];
        size_t seed_count = 0U;
        long count = 0L;
        curve->top1_min = 0.0;
        curve->top1_max = 0.0;
        for (size_t row_index = 0U; row_index < rows->count; ++row_index) {
            const MTReportRow *row = &rows->items[row_index];
            if (mt_tiny_curve_matches(curve, row)) {
                if (count == 0L || row->top1 < curve->top1_min) {
                    curve->top1_min = row->top1;
                }
                if (count == 0L || row->top1 > curve->top1_max) {
                    curve->top1_max = row->top1;
                }
                ++count;
                top1_sum += row->top1;
                top1_sq += row->top1 * row->top1;
                nll_sum += row->nll;
                nll_sq += row->nll * row->nll;
                ece_sum += row->ece;
                ece_sq += row->ece * row->ece;
                params_sum += row->parameters;
                flops_sum += row->forward_flops;
                if (!mt_seed_seen(seeds, seed_count, row->seed) &&
                    seed_count < sizeof(seeds) / sizeof(seeds[0])) {
                    seeds[seed_count++] = row->seed;
                }
            }
        }
        curve->rows = count;
        curve->seeds = (long)seed_count;
        if (count > 0L) {
            curve->top1_mean = top1_sum / (double)count;
            curve->top1_stdev = mt_report_stdev(top1_sum, top1_sq, count);
            curve->nll_mean = nll_sum / (double)count;
            curve->nll_stdev = mt_report_stdev(nll_sum, nll_sq, count);
            curve->ece_mean = ece_sum / (double)count;
            curve->ece_stdev = mt_report_stdev(ece_sum, ece_sq, count);
            curve->parameters_mean = params_sum / (double)count;
            curve->forward_flops_mean = flops_sum / (double)count;
        }
    }
    *curves_out = curves;
    *curve_count_out = curve_count;
    return 0;
}

static int mt_tiny_same_group(const MTTinyCurve *curve,
                              const char *hidden,
                              const char *domain,
                              const char *model_kind)
{
    return strcmp(curve->hidden, hidden) == 0 &&
           strcmp(curve->domain, domain) == 0 &&
           strcmp(curve->model_kind, model_kind) == 0;
}

static void mt_tiny_sort_curve_group(const MTTinyCurve **group, size_t count)
{
    for (size_t i = 0U; i < count; ++i) {
        for (size_t j = i + 1U; j < count; ++j) {
            if (group[j]->epochs < group[i]->epochs) {
                const MTTinyCurve *tmp = group[i];
                group[i] = group[j];
                group[j] = tmp;
            }
        }
    }
}

static int mt_tiny_plateau_group(const MTTinyCurve *curves,
                                 size_t curve_count,
                                 const char *hidden,
                                 const char *domain,
                                 const char *model_kind,
                                 double threshold,
                                 size_t plateau_window,
                                 MTTinyPlateau *out)
{
    const MTTinyCurve **group =
        (const MTTinyCurve **)calloc(curve_count == 0U ? 1U : curve_count, sizeof(*group));
    size_t count = 0U;
    size_t window;
    const MTTinyCurve *first;
    const MTTinyCurve *last;
    const MTTinyCurve *prev;
    const MTTinyCurve *recent_first;
    const MTTinyCurve *best;
    if (group == NULL) {
        return -1;
    }
    for (size_t i = 0U; i < curve_count; ++i) {
        if (mt_tiny_same_group(&curves[i], hidden, domain, model_kind)) {
            group[count++] = &curves[i];
        }
    }
    if (count == 0U) {
        free(group);
        return 0;
    }
    mt_tiny_sort_curve_group(group, count);
    window = plateau_window == 0U ? 1U : plateau_window;
    if (window > count) {
        window = count;
    }
    first = group[0];
    last = group[count - 1U];
    prev = count > 1U ? group[count - 2U] : last;
    recent_first = group[count - window];
    best = first;
    for (size_t i = 1U; i < count; ++i) {
        if (group[i]->top1_mean > best->top1_mean) {
            best = group[i];
        }
    }
    memset(out, 0, sizeof(*out));
    out->present = 1;
    snprintf(out->hidden, sizeof(out->hidden), "%s", hidden);
    out->domain = domain;
    out->model_kind = model_kind;
    out->best_epochs = best->epochs;
    out->best_top1_mean = best->top1_mean;
    out->best_top1_stdev = best->top1_stdev;
    out->last_epochs = last->epochs;
    out->last_top1_mean = last->top1_mean;
    out->last_gain = last->top1_mean - prev->top1_mean;
    out->last_window_gain = last->top1_mean - recent_first->top1_mean;
    out->best_gain_from_first = best->top1_mean - first->top1_mean;
    out->plateau_threshold = threshold;
    out->plateau_window = window;
    out->best_epoch_in_last_window = best->epochs >= recent_first->epochs;
    out->plateau_detected = fabs(out->last_gain) <= threshold &&
                            fabs(out->last_window_gain) <= threshold &&
                            out->best_epoch_in_last_window;
    out->parameters_mean = last->parameters_mean;
    out->forward_flops_mean = last->forward_flops_mean;
    free(group);
    return 1;
}

static int mt_tiny_collect_plateaus(const MTTinyCurve *curves,
                                    size_t curve_count,
                                    double threshold,
                                    size_t plateau_window,
                                    MTTinyPlateau **plateaus_out,
                                    size_t *plateau_count_out)
{
    MTTinyPlateau *plateaus =
        (MTTinyPlateau *)calloc(curve_count == 0U ? 1U : curve_count, sizeof(*plateaus));
    size_t plateau_count = 0U;
    if (plateaus == NULL) {
        return -1;
    }
    for (size_t i = 0U; i < curve_count; ++i) {
        int seen = 0;
        for (size_t j = 0U; j < plateau_count; ++j) {
            if (strcmp(plateaus[j].hidden, curves[i].hidden) == 0 &&
                strcmp(plateaus[j].domain, curves[i].domain) == 0 &&
                strcmp(plateaus[j].model_kind, curves[i].model_kind) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            int rc = mt_tiny_plateau_group(curves,
                                           curve_count,
                                           curves[i].hidden,
                                           curves[i].domain,
                                           curves[i].model_kind,
                                           threshold,
                                           plateau_window,
                                           &plateaus[plateau_count]);
            if (rc < 0) {
                free(plateaus);
                return -1;
            }
            if (rc > 0) {
                ++plateau_count;
            }
        }
    }
    *plateaus_out = plateaus;
    *plateau_count_out = plateau_count;
    return 0;
}

static const MTTinyPlateau *mt_tiny_find_plateau(const MTTinyPlateau *plateaus,
                                                 size_t plateau_count,
                                                 const char *hidden,
                                                 const char *domain,
                                                 const char *model_kind)
{
    for (size_t i = 0U; i < plateau_count; ++i) {
        if (plateaus[i].present &&
            strcmp(plateaus[i].hidden, hidden) == 0 &&
            strcmp(plateaus[i].domain, domain) == 0 &&
            strcmp(plateaus[i].model_kind, model_kind) == 0) {
            return &plateaus[i];
        }
    }
    return NULL;
}

static int mt_tiny_collect_widths(const MTTinyPlateau *plateaus,
                                  size_t plateau_count,
                                  const MTTinyCapacity *capacities,
                                  size_t capacity_count,
                                  const char *const *domains,
                                  size_t domain_count,
                                  const char *const *models,
                                  size_t model_count,
                                  MTTinyWidth **widths_out,
                                  size_t *width_count_out)
{
    size_t capacity = domain_count * model_count;
    MTTinyWidth *widths = (MTTinyWidth *)calloc(capacity == 0U ? 1U : capacity, sizeof(*widths));
    size_t width_count = 0U;
    if (widths == NULL) {
        return -1;
    }
    for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
        for (size_t model_index = 0U; model_index < model_count; ++model_index) {
            const MTTinyPlateau *first = NULL;
            const MTTinyPlateau *last = NULL;
            const MTTinyPlateau *best = NULL;
            for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
                const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                                plateau_count,
                                                                capacities[cap_index].label,
                                                                domains[domain_index],
                                                                models[model_index]);
                if (row == NULL) {
                    continue;
                }
                if (first == NULL) {
                    first = row;
                }
                last = row;
                if (best == NULL || row->best_top1_mean > best->best_top1_mean) {
                    best = row;
                }
            }
            if (first != NULL && last != NULL && best != NULL) {
                MTTinyWidth *width = &widths[width_count++];
                width->present = 1;
                width->domain = domains[domain_index];
                width->model_kind = models[model_index];
                snprintf(width->smallest_hidden, sizeof(width->smallest_hidden), "%s", first->hidden);
                snprintf(width->largest_hidden, sizeof(width->largest_hidden), "%s", last->hidden);
                snprintf(width->best_hidden, sizeof(width->best_hidden), "%s", best->hidden);
                width->best_top1_mean = best->best_top1_mean;
                width->largest_minus_smallest_top1 =
                    last->best_top1_mean - first->best_top1_mean;
                width->larger_width_helped = strcmp(best->hidden, first->hidden) != 0;
            }
        }
    }
    *widths_out = widths;
    *width_count_out = width_count;
    return 0;
}

static const MTTinyWidth *mt_tiny_find_width(const MTTinyWidth *widths,
                                             size_t width_count,
                                             const char *domain,
                                             const char *model_kind)
{
    for (size_t i = 0U; i < width_count; ++i) {
        if (widths[i].present &&
            strcmp(widths[i].domain, domain) == 0 &&
            strcmp(widths[i].model_kind, model_kind) == 0) {
            return &widths[i];
        }
    }
    return NULL;
}

static const MTTinyCurve *mt_tiny_find_curve(const MTTinyCurve *curves,
                                             size_t curve_count,
                                             const char *hidden,
                                             const char *domain,
                                             const char *model_kind,
                                             size_t epochs)
{
    for (size_t i = 0U; i < curve_count; ++i) {
        if (curves[i].present &&
            curves[i].epochs == epochs &&
            strcmp(curves[i].hidden, hidden) == 0 &&
            strcmp(curves[i].domain, domain) == 0 &&
            strcmp(curves[i].model_kind, model_kind) == 0) {
            return &curves[i];
        }
    }
    return NULL;
}

static int mt_tiny_collect_verdicts(const MTTinyPlateau *plateaus,
                                    size_t plateau_count,
                                    const MTTinyWidth *widths,
                                    size_t width_count,
                                    const MTTinyCapacity *capacities,
                                    size_t capacity_count,
                                    const uint64_t *seeds,
                                    size_t seed_count,
                                    const size_t *epoch_grid,
                                    size_t epoch_count,
                                    const char *const *domains,
                                    size_t domain_count,
                                    const char *const *models,
                                    size_t model_count,
                                    MTTinyVerdict **verdicts_out,
                                    size_t *verdict_count_out)
{
    size_t capacity = domain_count * model_count;
    MTTinyVerdict *verdicts =
        (MTTinyVerdict *)calloc(capacity == 0U ? 1U : capacity, sizeof(*verdicts));
    size_t verdict_count = 0U;
    if (verdicts == NULL) {
        return -1;
    }
    for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
        for (size_t model_index = 0U; model_index < model_count; ++model_index) {
            const MTTinyWidth *width =
                mt_tiny_find_width(widths, width_count, domains[domain_index], models[model_index]);
            MTTinyVerdict *verdict = &verdicts[verdict_count++];
            long group_count = 0L;
            long plateau_ready_count = 0L;
            int coverage_complete = 1;
            verdict->present = 1;
            verdict->domain = domains[domain_index];
            verdict->model_kind = models[model_index];
            verdict->restart_count = (long)seed_count;
            verdict->epoch_checkpoint_count = (long)epoch_count;
            for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
                const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                                plateau_count,
                                                                capacities[cap_index].label,
                                                                domains[domain_index],
                                                                models[model_index]);
                if (row == NULL) {
                    coverage_complete = 0;
                    continue;
                }
                ++group_count;
                if (row->plateau_detected) {
                    ++plateau_ready_count;
                }
            }
            verdict->capacity_count = group_count;
            verdict->capacity_coverage_complete = coverage_complete;
            verdict->plateau_count = plateau_ready_count;
            verdict->all_plateau_detected = group_count > 0L && plateau_ready_count == group_count;
            verdict->width_verdict_present = width != NULL;
            if (width != NULL) {
                snprintf(verdict->best_hidden, sizeof(verdict->best_hidden), "%s", width->best_hidden);
                verdict->best_top1_mean = width->best_top1_mean;
                verdict->larger_width_helped = width->larger_width_helped;
            }
            verdict->audit_ready = verdict->capacity_coverage_complete &&
                                   verdict->all_plateau_detected &&
                                   verdict->width_verdict_present;
        }
    }
    (void)seeds;
    (void)epoch_grid;
    *verdicts_out = verdicts;
    *verdict_count_out = verdict_count;
    return 0;
}

static int mt_write_tiny_learning_curve_csv(const char *path,
                                            const MTTinyCurve *curves,
                                            size_t count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("hidden,domain,model_kind,epochs,seeds,top1_mean,top1_stdev,top1_min,top1_max,nll_mean,nll_stdev,ece_mean,ece_stdev,parameters_mean,forward_flops_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyCurve *row = &curves[i];
        fprintf(fp, "%s,%s,%s,%zu,%ld,", row->hidden, row->domain, row->model_kind, row->epochs, row->seeds);
        mt_csv_double(fp, row->top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->top1_stdev); fputc(',', fp);
        mt_csv_double(fp, row->top1_min); fputc(',', fp);
        mt_csv_double(fp, row->top1_max); fputc(',', fp);
        mt_csv_double(fp, row->nll_mean); fputc(',', fp);
        mt_csv_double(fp, row->nll_stdev); fputc(',', fp);
        mt_csv_double(fp, row->ece_mean); fputc(',', fp);
        mt_csv_double(fp, row->ece_stdev); fputc(',', fp);
        mt_csv_double(fp, row->parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->forward_flops_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_plateau_csv(const char *path,
                                     const MTTinyPlateau *plateaus,
                                     size_t count)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    fputs("schema,report_version,date,hidden,domain,model_kind,best_epochs,best_top1_mean,best_top1_stdev,last_epochs,last_top1_mean,last_gain,last_window_gain,best_gain_from_first,plateau_threshold,plateau_window,plateau_method,best_epoch_in_last_window,plateau_detected,parameters_mean,forward_flops_mean\n",
          fp);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyPlateau *row = &plateaus[i];
        fprintf(fp,
                "crlp.multitask.tiny_net_plateau.v1,2,%s,%s,%s,%s,%zu,",
                date,
                row->hidden,
                row->domain,
                row->model_kind,
                row->best_epochs);
        mt_csv_double(fp, row->best_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->best_top1_stdev); fprintf(fp, ",%zu,", row->last_epochs);
        mt_csv_double(fp, row->last_top1_mean); fputc(',', fp);
        mt_csv_double(fp, row->last_gain); fputc(',', fp);
        mt_csv_double(fp, row->last_window_gain); fputc(',', fp);
        mt_csv_double(fp, row->best_gain_from_first); fputc(',', fp);
        mt_csv_double(fp, row->plateau_threshold);
        fprintf(fp,
                ",%zu,last_step_and_window_gain,%s,%s,",
                row->plateau_window,
                row->best_epoch_in_last_window ? "true" : "false",
                row->plateau_detected ? "true" : "false");
        mt_csv_double(fp, row->parameters_mean); fputc(',', fp);
        mt_csv_double(fp, row->forward_flops_mean); fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_plateau_jsonl(const char *path,
                                       const MTTinyPlateau *plateaus,
                                       size_t count)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyPlateau *row = &plateaus[i];
        fputs("{\"schema\":\"crlp.multitask.tiny_net_plateau.v1\",\"report_version\":2,\"date\":", fp);
        mt_json_write_string(fp, date);
        fputs(",\"hidden\":", fp); mt_json_write_string(fp, row->hidden);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, row->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, row->model_kind);
        fprintf(fp, ",\"best_epochs\":%zu,\"best_top1_mean\":", row->best_epochs);
        mt_json_number_or_null(fp, row->best_top1_mean);
        fputs(",\"best_top1_stdev\":", fp); mt_json_number_or_null(fp, row->best_top1_stdev);
        fprintf(fp, ",\"last_epochs\":%zu,\"last_top1_mean\":", row->last_epochs);
        mt_json_number_or_null(fp, row->last_top1_mean);
        fputs(",\"last_gain\":", fp); mt_json_number_or_null(fp, row->last_gain);
        fputs(",\"last_window_gain\":", fp); mt_json_number_or_null(fp, row->last_window_gain);
        fputs(",\"best_gain_from_first\":", fp); mt_json_number_or_null(fp, row->best_gain_from_first);
        fputs(",\"plateau_threshold\":", fp); mt_json_number_or_null(fp, row->plateau_threshold);
        fprintf(fp,
                ",\"plateau_window\":%zu,\"plateau_method\":\"last_step_and_window_gain\","
                "\"best_epoch_in_last_window\":%s,\"plateau_detected\":%s,"
                "\"parameters_mean\":",
                row->plateau_window,
                row->best_epoch_in_last_window ? "true" : "false",
                row->plateau_detected ? "true" : "false");
        mt_json_number_or_null(fp, row->parameters_mean);
        fputs(",\"forward_flops_mean\":", fp); mt_json_number_or_null(fp, row->forward_flops_mean);
        fputs("}\n", fp);
    }
    fclose(fp);
    return 0;
}

static void mt_write_capacity_ladder_json(FILE *fp,
                                          const MTTinyCapacity *capacities,
                                          size_t count)
{
    fputc('[', fp);
    for (size_t i = 0U; i < count; ++i) {
        if (i != 0U) {
            fputc(',', fp);
        }
        mt_json_write_string(fp, capacities[i].label);
    }
    fputc(']', fp);
}

static int mt_write_tiny_width_jsonl(const char *path,
                                     const MTTinyWidth *widths,
                                     size_t count,
                                     const MTTinyCapacity *capacities,
                                     size_t capacity_count)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyWidth *row = &widths[i];
        fputs("{\"schema\":\"crlp.multitask.tiny_net_width_help.v1\",\"report_version\":1,\"date\":", fp);
        mt_json_write_string(fp, date);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, row->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, row->model_kind);
        fputs(",\"capacity_ladder\":", fp); mt_write_capacity_ladder_json(fp, capacities, capacity_count);
        fputs(",\"smallest_hidden\":", fp); mt_json_write_string(fp, row->smallest_hidden);
        fputs(",\"largest_hidden\":", fp); mt_json_write_string(fp, row->largest_hidden);
        fputs(",\"best_hidden\":", fp); mt_json_write_string(fp, row->best_hidden);
        fputs(",\"best_top1_mean\":", fp); mt_json_number_or_null(fp, row->best_top1_mean);
        fputs(",\"largest_minus_smallest_top1\":", fp); mt_json_number_or_null(fp, row->largest_minus_smallest_top1);
        fprintf(fp, ",\"larger_width_helped\":%s}\n", row->larger_width_helped ? "true" : "false");
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_verdict_jsonl(const char *path,
                                       const MTTinyVerdict *verdicts,
                                       size_t count,
                                       const MTTinyCapacity *capacities,
                                       size_t capacity_count,
                                       const size_t *epoch_grid,
                                       size_t epoch_count,
                                       double plateau_threshold,
                                       size_t plateau_window)
{
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < count; ++i) {
        const MTTinyVerdict *row = &verdicts[i];
        fputs("{\"schema\":\"crlp.multitask.tiny_net_audit_verdict.v1\",\"report_version\":1,\"date\":", fp);
        mt_json_write_string(fp, date);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, row->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, row->model_kind);
        fputs(",\"capacity_ladder\":", fp); mt_write_capacity_ladder_json(fp, capacities, capacity_count);
        fprintf(fp,
                ",\"capacity_count\":%ld,\"capacity_coverage_complete\":%s,"
                "\"restart_count\":%ld,\"epoch_checkpoints\":[",
                row->capacity_count,
                row->capacity_coverage_complete ? "true" : "false",
                row->restart_count);
        for (size_t j = 0U; j < epoch_count; ++j) {
            if (j != 0U) {
                fputc(',', fp);
            }
            fprintf(fp, "%zu", epoch_grid[j]);
        }
        fprintf(fp,
                "],\"epoch_checkpoint_count\":%ld,\"plateau_threshold\":%.17g,"
                "\"plateau_window\":%zu,\"plateau_count\":%ld,"
                "\"all_plateau_detected\":%s,\"width_verdict_present\":%s,"
                "\"best_hidden\":",
                row->epoch_checkpoint_count,
                plateau_threshold,
                plateau_window,
                row->plateau_count,
                row->all_plateau_detected ? "true" : "false",
                row->width_verdict_present ? "true" : "false");
        mt_json_write_string(fp, row->best_hidden[0] == '\0' ? "" : row->best_hidden);
        fputs(",\"best_top1_mean\":", fp); mt_json_number_or_null(fp, row->best_top1_mean);
        fprintf(fp,
                ",\"larger_width_helped\":%s,\"audit_ready\":%s}\n",
                row->larger_width_helped ? "true" : "false",
                row->audit_ready ? "true" : "false");
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_diagnostics_jsonl(const char *path,
                                           const MTTinyCurve *curves,
                                           size_t curve_count,
                                           const MTTinyPlateau *plateaus,
                                           size_t plateau_count,
                                           const MTTinyWidth *widths,
                                           size_t width_count,
                                           const MTTinyVerdict *verdicts,
                                           size_t verdict_count,
                                           const MTTinyCapacity *capacities,
                                           size_t capacity_count)
{
    const double seed_stdev_threshold = 0.03;
    char date[11];
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    for (size_t i = 0U; i < verdict_count; ++i) {
        const MTTinyVerdict *verdict = &verdicts[i];
        const MTTinyWidth *width =
            mt_tiny_find_width(widths, width_count, verdict->domain, verdict->model_kind);
        const MTTinyPlateau *best = NULL;
        const MTTinyPlateau *smallest = NULL;
        const MTTinyPlateau *largest = NULL;
        const char *diagnosis = "mixed_unready";
        const char *next_step = "inspect_learning_curves";
        long observed_capacity_count = 0L;
        long non_plateau_count = 0L;
        long late_best_not_plateau_count = 0L;
        long high_seed_stdev_count = 0L;
        double max_seed_stdev = 0.0;
        double max_best_seed_spread = 0.0;
        double max_abs_last_gain = 0.0;
        double max_abs_last_window_gain = 0.0;
        double width_delta = 0.0;

        for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
            const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                            plateau_count,
                                                            capacities[cap_index].label,
                                                            verdict->domain,
                                                            verdict->model_kind);
            if (row == NULL) {
                continue;
            }
            ++observed_capacity_count;
            if (smallest == NULL) {
                smallest = row;
            }
            largest = row;
            if (best == NULL || row->best_top1_mean > best->best_top1_mean) {
                best = row;
            }
            if (!row->plateau_detected) {
                ++non_plateau_count;
                if (row->best_epoch_in_last_window) {
                    ++late_best_not_plateau_count;
                }
            }
            if (row->best_top1_stdev >= seed_stdev_threshold) {
                ++high_seed_stdev_count;
            }
            if (row->best_top1_stdev > max_seed_stdev) {
                max_seed_stdev = row->best_top1_stdev;
            }
            if (fabs(row->last_gain) > max_abs_last_gain) {
                max_abs_last_gain = fabs(row->last_gain);
            }
            if (fabs(row->last_window_gain) > max_abs_last_window_gain) {
                max_abs_last_window_gain = fabs(row->last_window_gain);
            }
            {
                const MTTinyCurve *curve = mt_tiny_find_curve(curves,
                                                              curve_count,
                                                              row->hidden,
                                                              row->domain,
                                                              row->model_kind,
                                                              row->best_epochs);
                if (curve != NULL) {
                    double spread = curve->top1_max - curve->top1_min;
                    if (spread > max_best_seed_spread) {
                        max_best_seed_spread = spread;
                    }
                }
            }
        }

        if (width != NULL) {
            width_delta = width->largest_minus_smallest_top1;
        }
        if (verdict->audit_ready) {
            diagnosis = "audit_ready";
            next_step = "eligible_for_claim_review";
        } else if (best == NULL) {
            diagnosis = "missing_coverage";
            next_step = "finish_or_resume_audit_cells";
        } else if (best->best_top1_mean < 0.25) {
            diagnosis = "weak_signal";
            next_step = "inspect_feature_or_label_encoding";
        } else if (high_seed_stdev_count > 0L &&
                   max_seed_stdev >= fabs(width_delta)) {
            diagnosis = "seed_variance_dominates_width";
            next_step = "increase_restarts_before_width_claim";
        } else if (observed_capacity_count > 0L &&
                   non_plateau_count == observed_capacity_count) {
            diagnosis = "not_plateaued";
            next_step = "extend_or_stabilize_training";
        } else if (width != NULL && !width->larger_width_helped) {
            diagnosis = "not_width_limited";
            next_step = "try_protocol_artifacts_or_sampling";
        } else if (width != NULL && width->larger_width_helped &&
                   non_plateau_count > 0L) {
            diagnosis = "capacity_helped_but_unsettled";
            next_step = "run_stronger_baseline_ladder";
        }

        fputs("{\"schema\":\"crlp.multitask.tiny_net_diagnostic.v1\","
              "\"report_version\":1,\"date\":",
              fp);
        mt_json_write_string(fp, date);
        fputs(",\"domain\":", fp); mt_json_write_string(fp, verdict->domain);
        fputs(",\"model_kind\":", fp); mt_json_write_string(fp, verdict->model_kind);
        fputs(",\"capacity_ladder\":", fp); mt_write_capacity_ladder_json(fp, capacities, capacity_count);
        fprintf(fp,
                ",\"capacity_count\":%ld,\"observed_capacity_count\":%ld,"
                "\"restart_count\":%ld,\"epoch_checkpoint_count\":%ld,"
                "\"plateau_count\":%ld,\"non_plateau_count\":%ld,"
                "\"late_best_not_plateau_count\":%ld,"
                "\"high_seed_stdev_count\":%ld,\"audit_ready\":%s,"
                "\"best_hidden\":",
                verdict->capacity_count,
                observed_capacity_count,
                verdict->restart_count,
                verdict->epoch_checkpoint_count,
                verdict->plateau_count,
                non_plateau_count,
                late_best_not_plateau_count,
                high_seed_stdev_count,
                verdict->audit_ready ? "true" : "false");
        mt_json_write_string(fp, best == NULL ? "" : best->hidden);
        if (best != NULL) {
            fprintf(fp, ",\"best_epochs\":%zu,\"best_top1_mean\":", best->best_epochs);
            mt_json_number_or_null(fp, best->best_top1_mean);
            fputs(",\"best_top1_stdev\":", fp); mt_json_number_or_null(fp, best->best_top1_stdev);
        } else {
            fputs(",\"best_epochs\":null,\"best_top1_mean\":null,\"best_top1_stdev\":null", fp);
        }
        fputs(",\"smallest_hidden\":", fp);
        mt_json_write_string(fp, smallest == NULL ? "" : smallest->hidden);
        fputs(",\"smallest_best_top1_mean\":", fp);
        mt_json_number_or_null(fp, smallest == NULL ? NAN : smallest->best_top1_mean);
        fputs(",\"largest_hidden\":", fp);
        mt_json_write_string(fp, largest == NULL ? "" : largest->hidden);
        fputs(",\"largest_best_top1_mean\":", fp);
        mt_json_number_or_null(fp, largest == NULL ? NAN : largest->best_top1_mean);
        fputs(",\"largest_minus_smallest_top1\":", fp);
        mt_json_number_or_null(fp, width == NULL ? NAN : width->largest_minus_smallest_top1);
        fprintf(fp, ",\"larger_width_helped\":%s,\"max_seed_stdev\":",
                width != NULL && width->larger_width_helped ? "true" : "false");
        mt_json_number_or_null(fp, max_seed_stdev);
        fputs(",\"max_best_seed_spread\":", fp); mt_json_number_or_null(fp, max_best_seed_spread);
        fputs(",\"max_abs_last_gain\":", fp); mt_json_number_or_null(fp, max_abs_last_gain);
        fputs(",\"max_abs_last_window_gain\":", fp); mt_json_number_or_null(fp, max_abs_last_window_gain);
        fputs(",\"seed_stdev_threshold\":", fp); mt_json_number_or_null(fp, seed_stdev_threshold);
        fputs(",\"diagnosis\":", fp); mt_json_write_string(fp, diagnosis);
        fputs(",\"next_step\":", fp); mt_json_write_string(fp, next_step);
        fputs("}\n", fp);
    }
    fclose(fp);
    return 0;
}

static int mt_write_tiny_report_md(const char *path,
                                   const uint64_t *seeds,
                                   size_t seed_count,
                                   const MTTinyCapacity *capacities,
                                   size_t capacity_count,
                                   const size_t *epoch_grid,
                                   size_t epoch_count,
                                   const MTConfig *base,
                                   const MTTinyPlateau *plateaus,
                                   size_t plateau_count,
                                   const MTTinyWidth *widths,
                                   size_t width_count,
                                   const MTTinyVerdict *verdicts,
                                   size_t verdict_count,
                                   const char *const *domains,
                                   size_t domain_count,
                                   const char *const *models,
                                   size_t model_count)
{
    FILE *fp = fopen(path, "w");
    char date[11];
    if (fp == NULL) {
        return -1;
    }
    mt_today_iso(date);
    fprintf(fp, "# Fully Trained Tiny-Net Audit\n\nDate: %s\nSeeds: `", date);
    for (size_t i = 0U; i < seed_count; ++i) {
        fprintf(fp, "%s%llu", i == 0U ? "" : ", ", (unsigned long long)seeds[i]);
    }
    fputs("`\nCapacities: `", fp);
    for (size_t i = 0U; i < capacity_count; ++i) {
        fprintf(fp, "%s%s", i == 0U ? "" : ", ", capacities[i].label);
    }
    fputs("`\nModel variants: `", fp);
    for (size_t i = 0U; i < model_count; ++i) {
        fprintf(fp, "%s%s", i == 0U ? "" : ", ", models[i]);
    }
    fputs("`\nEpoch checkpoints: `", fp);
    for (size_t i = 0U; i < epoch_count; ++i) {
        fprintf(fp, "%s%zu", i == 0U ? "" : ", ", epoch_grid[i]);
    }
    fprintf(fp,
            "`\nSteps per epoch: `%zu`; eval samples: `%zu`\n\n"
            "## Best Checkpoints\n\n"
            "| Hidden | Model | Prime | Grid | Delayed | ARC |\n"
            "| --- | --- | ---: | ---: | ---: | ---: |\n",
            base->steps_per_epoch,
            base->eval_samples);
    for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
        for (size_t model_index = 0U; model_index < model_count; ++model_index) {
            fprintf(fp, "| `%s` | `%s` |", capacities[cap_index].label, models[model_index]);
            for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
                const MTTinyPlateau *row = mt_tiny_find_plateau(plateaus,
                                                                plateau_count,
                                                                capacities[cap_index].label,
                                                                domains[domain_index],
                                                                models[model_index]);
                char pct[32];
                if (row == NULL) {
                    fputs(" - |", fp);
                } else {
                    mt_format_pct(pct, sizeof(pct), row->best_top1_mean);
                    fprintf(fp, " %s @ %zu |", pct, row->best_epochs);
                }
            }
            fputc('\n', fp);
        }
    }
    fputs("\n## Plateau Flags\n\n"
          "| Hidden | Domain | Model | Last gain | Window gain | Plateau |\n"
          "| --- | --- | --- | ---: | ---: | --- |\n",
          fp);
    for (size_t i = 0U; i < plateau_count; ++i) {
        char gain[32], window_gain[32];
        mt_format_delta(gain, sizeof(gain), plateaus[i].last_gain);
        mt_format_delta(window_gain, sizeof(window_gain), plateaus[i].last_window_gain);
        fprintf(fp,
                "| `%s` | `%s` | `%s` | %s | %s | %s |\n",
                plateaus[i].hidden,
                plateaus[i].domain,
                plateaus[i].model_kind,
                gain,
                window_gain,
                plateaus[i].plateau_detected ? "yes" : "no");
    }
    fputs("\n## Width Verdict\n\n"
          "| Domain | Model | Best hidden | Largest-smallest top-1 | Larger helped |\n"
          "| --- | --- | ---: | ---: | --- |\n",
          fp);
    for (size_t i = 0U; i < width_count; ++i) {
        char delta[32];
        mt_format_delta(delta, sizeof(delta), widths[i].largest_minus_smallest_top1);
        fprintf(fp,
                "| `%s` | `%s` | `%s` | %s | %s |\n",
                widths[i].domain,
                widths[i].model_kind,
                widths[i].best_hidden,
                delta,
                widths[i].larger_width_helped ? "yes" : "no");
    }
    fputs("\n## Audit Verdict\n\n"
          "| Domain | Model | Coverage | Plateau rows | Best hidden | Audit ready |\n"
          "| --- | --- | --- | ---: | ---: | --- |\n",
          fp);
    for (size_t i = 0U; i < verdict_count; ++i) {
        fprintf(fp,
                "| `%s` | `%s` | %s | %ld/%ld | `%s` | %s |\n",
                verdicts[i].domain,
                verdicts[i].model_kind,
                verdicts[i].capacity_coverage_complete ? "yes" : "no",
                verdicts[i].plateau_count,
                verdicts[i].capacity_count,
                verdicts[i].best_hidden[0] == '\0' ? "-" : verdicts[i].best_hidden,
                verdicts[i].audit_ready ? "yes" : "no");
    }
    fclose(fp);
    return 0;
}

static size_t mt_epoch_grid_max(const size_t *epoch_grid, size_t epoch_count)
{
    size_t max_epoch = 0U;
    for (size_t i = 0U; i < epoch_count; ++i) {
        if (epoch_grid[i] > max_epoch) {
            max_epoch = epoch_grid[i];
        }
    }
    return max_epoch;
}

static int mt_epoch_grid_contains(const size_t *epoch_grid,
                                  size_t epoch_count,
                                  size_t epoch)
{
    for (size_t i = 0U; i < epoch_count; ++i) {
        if (epoch_grid[i] == epoch) {
            return 1;
        }
    }
    return 0;
}

static void mt_tiny_record_task_checkpoint(MTScorer *scorer,
                                           const MTData *data,
                                           const MTConfig *run,
                                           const char *model_kind,
                                           const char *training_order,
                                           const size_t updates[MT_TASK_COUNT],
                                           size_t epoch,
                                           MTTask task)
{
    MTConfig checkpoint = *run;
    checkpoint.epochs = epoch;
    MTMetrics metrics =
        mt_evaluate_task(scorer, data, &checkpoint, task, checkpoint.eval_samples);
    mt_print_metrics_json(scorer,
                          model_kind,
                          training_order,
                          task,
                          updates[task],
                          &checkpoint,
                          metrics);
    if (task == MT_TASK_ARC_AGI) {
        mt_print_arc_ood_task_breakdown(scorer,
                                        data,
                                        &checkpoint,
                                        model_kind,
                                        training_order,
                                        updates[task]);
    }
}

static void mt_tiny_record_checkpoint(MTScorer *scorer,
                                      const MTData *data,
                                      const MTConfig *run,
                                      const char *model_kind,
                                      const char *training_order,
                                      const size_t updates[MT_TASK_COUNT],
                                      size_t epoch)
{
    MTConfig checkpoint = *run;
    MTTask eval_tasks[MT_TASK_COUNT];
    size_t eval_count;
    checkpoint.epochs = epoch;
    eval_count = mt_fill_active_tasks(checkpoint.eval_task_mask,
                                      eval_tasks,
                                      MT_TASK_COUNT);
    for (size_t task_index = 0U; task_index < eval_count; ++task_index) {
        MTTask task = eval_tasks[task_index];
        mt_tiny_record_task_checkpoint(scorer,
                                       data,
                                       &checkpoint,
                                       model_kind,
                                       training_order,
                                       updates,
                                       epoch,
                                       task);
    }
}

static int mt_tiny_cell_path(char *out,
                             size_t out_size,
                             const char *cell_dir,
                             const MTTinyCapacity *capacity,
                             uint64_t seed)
{
    char filename[128];
    int written;
    if (out == NULL || cell_dir == NULL || capacity == NULL) {
        return -1;
    }
    written = snprintf(filename,
                       sizeof(filename),
                       "tiny_net_plateau.%s.seed-%llu.jsonl",
                       capacity->label,
                       (unsigned long long)seed);
    if (written < 0 || (size_t)written >= sizeof(filename)) {
        return -1;
    }
    return mt_join_path(out, out_size, cell_dir, filename);
}

static uint64_t mt_fnv1a_update_bytes(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0U; i < size; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t mt_fnv1a_update_text(uint64_t hash, const char *text)
{
    const char *value = text == NULL ? "" : text;
    return mt_fnv1a_update_bytes(hash, value, strlen(value));
}

static uint64_t mt_tiny_audit_signature(const MTConfig *base,
                                        const size_t *epoch_grid,
                                        size_t epoch_count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    char buffer[128];
    char task_set[128];
    char eval_task_set[128];
    mt_task_set_name(base->task_mask, task_set, sizeof(task_set));
    mt_task_set_name(base->eval_task_mask, eval_task_set, sizeof(eval_task_set));
    hash = mt_fnv1a_update_text(hash, "tiny-net-plateau-v2;");
    hash = mt_fnv1a_update_text(hash, task_set);
    hash = mt_fnv1a_update_text(hash, ";eval=");
    hash = mt_fnv1a_update_text(hash, eval_task_set);
    snprintf(buffer,
             sizeof(buffer),
             ";steps=%zu;eval_samples=%zu;lr=%.17g;order=%s;tags=%d;heads=%d;baselines=%d;",
             base->steps_per_epoch,
             base->eval_samples,
             (double)base->learning_rate,
             mt_training_order_name((MTTrainingOrder)base->training_order),
             base->use_task_tags,
             base->run_task_head_model,
             base->run_task_baselines);
    hash = mt_fnv1a_update_text(hash, buffer);
    for (size_t i = 0U; i < epoch_count; ++i) {
        snprintf(buffer, sizeof(buffer), "%zu,", epoch_grid[i]);
        hash = mt_fnv1a_update_text(hash, buffer);
    }
    return hash;
}

static int mt_tiny_shard_selected(size_t cell_index,
                                  size_t shard_index,
                                  size_t shard_count)
{
    if (shard_count <= 1U) {
        return 1;
    }
    return cell_index % shard_count == shard_index - 1U;
}

static int mt_run_tiny_net_plateau_cell(const MTConfig *base,
                                        const MTTinyCapacity *capacity,
                                        uint64_t seed,
                                        const size_t *epoch_grid,
                                        size_t epoch_count,
                                        MTReportRows *rows,
                                        FILE *raw)
{
    uint32_t all_task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                             mt_task_bit(MT_TASK_GRID_PATH) |
                             mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                             mt_task_bit(MT_TASK_ARC_AGI);
    size_t max_epoch;
    MTConfig run;
    MTData data;
    int data_ready = 0;
    int status = 0;
    nn_real *candidate_inputs = NULL;
    NeuralNet *shared = NULL;
    MTTaskHeadedScorer *task_heads = NULL;

    if (base == NULL || capacity == NULL || epoch_grid == NULL || epoch_count == 0U ||
        rows == NULL || raw == NULL) {
        return 0;
    }
    max_epoch = mt_epoch_grid_max(epoch_grid, epoch_count);
    if (max_epoch == 0U) {
        return 0;
    }
    run = *base;

    run.tiny_net_plateau_audit = 0;
    run.adapter_heads_report = 0;
    run.hidden1 = capacity->hidden1;
    run.hidden2 = capacity->hidden2;
    run.seed = seed;
    run.epochs = max_epoch;
    run.task_mask = all_task_mask;
    run.eval_task_mask = base->eval_task_mask == 0U ? all_task_mask : base->eval_task_mask;
    run.arc_eval_suite = MT_ARC_EVAL_0520FDE7;
    run.arc_ood_task = -1;
    run.report_arc_ood_tasks = 0;
    run.quiet = 1;
    run.roadmap_item = "E-MT.9";
    run.adapter_eval_suite = NULL;
    run.json_output = raw;
    run.report_rows = rows;

    if (!mt_prepare_data(&data)) {
        fputs("failed to prepare multitask datasets\n", stderr);
        goto done;
    }
    data_ready = 1;

    candidate_inputs =
        calloc((size_t)MT_MAX_CANDIDATES * MT_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        goto done;
    }

    shared = mt_create_model(&run, (unsigned int)(run.seed & 0xffffffffu));
    if (shared == NULL) {
        goto done;
    }
    {
        uint32_t rng =
            (uint32_t)(run.seed ^ (run.seed >> 32) ^ UINT64_C(0x31415926));
        size_t shared_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
        MTScorer shared_scorer;
        shared_scorer.kind = MT_SCORER_NN;
        shared_scorer.net = shared;
        shared_scorer.task_heads = NULL;
        for (size_t epoch = 0U; epoch < max_epoch; ++epoch) {
            if (!mt_train_mixed_one_epoch(shared,
                                          &data,
                                          &run,
                                          epoch,
                                          &rng,
                                          shared_updates,
                                          candidate_inputs)) {
                goto done;
            }
            if (mt_epoch_grid_contains(epoch_grid, epoch_count, epoch + 1U)) {
                mt_tiny_record_checkpoint(&shared_scorer,
                                          &data,
                                          &run,
                                          "shared",
                                          mt_training_order_name(
                                              (MTTrainingOrder)run.training_order),
                                          shared_updates,
                                          epoch + 1U);
            }
        }
    }

    if (run.run_task_head_model) {
        task_heads =
            mt_task_heads_create(&run,
                                 (unsigned int)((run.seed + UINT64_C(303)) &
                                                UINT64_C(0xffffffff)));
        if (task_heads == NULL) {
            goto done;
        }
        uint32_t rng =
            (uint32_t)(run.seed ^ (run.seed >> 32) ^ UINT64_C(0x27182818));
        size_t task_head_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
        MTScorer task_head_scorer;
        task_head_scorer.kind = MT_SCORER_TASK_HEADS;
        task_head_scorer.net = NULL;
        task_head_scorer.task_heads = task_heads;
        for (size_t epoch = 0U; epoch < max_epoch; ++epoch) {
            if (!mt_train_mixed_task_heads_one_epoch(task_heads,
                                                     &data,
                                                     &run,
                                                     epoch,
                                                     &rng,
                                                     task_head_updates,
                                                     candidate_inputs)) {
                goto done;
            }
            if (mt_epoch_grid_contains(epoch_grid, epoch_count, epoch + 1U)) {
                mt_tiny_record_checkpoint(&task_head_scorer,
                                          &data,
                                          &run,
                                          "shared_task_heads",
                                          mt_training_order_name(
                                              (MTTrainingOrder)run.training_order),
                                          task_head_updates,
                                          epoch + 1U);
            }
        }
        mt_task_heads_free(task_heads);
        task_heads = NULL;
    }

    if (run.run_task_baselines) {
        MTTask active_tasks[MT_TASK_COUNT];
        size_t active_count = mt_fill_active_tasks(run.task_mask,
                                                   active_tasks,
                                                   MT_TASK_COUNT);
        if (active_count == 0U) {
            goto done;
        }
        for (size_t task_index = 0U; task_index < active_count; ++task_index) {
            MTTask task = active_tasks[task_index];
            if ((run.eval_task_mask & mt_task_bit(task)) == 0) {
                continue;
            }
            NeuralNet *single =
                mt_create_model(&run,
                                (unsigned int)((run.seed + 101u + task_index) &
                                               0xffffffffu));
            if (single == NULL) {
                goto done;
            }
            uint32_t rng =
                (uint32_t)(run.seed ^ (run.seed >> 32) ^
                           (0x9e3779b9u + (uint32_t)task));
            size_t single_updates[MT_TASK_COUNT] = {0, 0, 0, 0};
            MTScorer single_scorer;
            single_scorer.kind = MT_SCORER_NN;
            single_scorer.net = single;
            single_scorer.task_heads = NULL;
            for (size_t epoch = 0U; epoch < max_epoch; ++epoch) {
                size_t target_updates = ((epoch + 1U) * run.steps_per_epoch) / active_count;
                if (target_updates == 0U) {
                    target_updates = 1U;
                }
                if (target_updates > single_updates[task]) {
                    double ignored_loss = 0.0;
                    if (!mt_train_single_task_steps(single,
                                                    &data,
                                                    &run,
                                                    task,
                                                    &rng,
                                                    candidate_inputs,
                                                    target_updates - single_updates[task],
                                                    &ignored_loss)) {
                        nn_free(single);
                        goto done;
                    }
                    single_updates[task] = target_updates;
                }
                if (mt_epoch_grid_contains(epoch_grid, epoch_count, epoch + 1U)) {
                    run.epochs = epoch + 1U;
                    mt_tiny_record_task_checkpoint(&single_scorer,
                                                   &data,
                                                   &run,
                                                   "single_task",
                                                   "domain_only_random",
                                                   single_updates,
                                                   epoch + 1U,
                                                   task);
                }
            }
            nn_free(single);
        }
    }

    status = 1;
done:
    mt_task_heads_free(task_heads);
    nn_free(shared);
    free(candidate_inputs);
    if (data_ready) {
        mt_free_data(&data);
    }
    return status;
}

static int mt_run_tiny_net_plateau_audit(const MTConfig *base)
{
    uint32_t all_task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                             mt_task_bit(MT_TASK_GRID_PATH) |
                             mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                             mt_task_bit(MT_TASK_ARC_AGI);
    MTTask audit_tasks[MT_TASK_COUNT];
    const char *domains[MT_TASK_COUNT];
    const char *models[3];
    size_t domain_count = 0U;
    size_t model_count = 0U;
    uint64_t seeds[128];
    MTTinyCapacity capacities[64];
    size_t epoch_grid[128];
    size_t seed_count = 0U, capacity_count = 0U, epoch_count = 0U;
    MTReportRows rows;
    MTTinyCurve *curves = NULL;
    MTTinyPlateau *plateaus = NULL;
    MTTinyWidth *widths = NULL;
    MTTinyVerdict *verdicts = NULL;
    size_t curve_count = 0U, plateau_count = 0U, width_count = 0U, verdict_count = 0U;
    size_t present_cell_count = 0U, missing_cell_count = 0U;
    char raw_path[4096], commands_path[4096], curve_path[4096], plateau_csv_path[4096];
    char plateau_jsonl_path[4096], width_path[4096], verdict_path[4096], markdown_path[4096];
    char diagnostics_path[4096];
    char cell_dir_path[4096], cell_dir_name[128];
    FILE *raw = NULL;
    FILE *commands = NULL;
    int status = EXIT_FAILURE;
    memset(&rows, 0, sizeof(rows));
    domain_count = mt_fill_active_tasks(base->eval_task_mask == 0U ? all_task_mask : base->eval_task_mask,
                                        audit_tasks,
                                        MT_TASK_COUNT);
    if (domain_count == 0U) {
        fputs("tiny-net audit has no eval domains\n", stderr);
        return EXIT_FAILURE;
    }
    for (size_t i = 0U; i < domain_count; ++i) {
        domains[i] = mt_task_name(audit_tasks[i]);
    }
    models[model_count++] = "shared";
    if (base->run_task_head_model) {
        models[model_count++] = "shared_task_heads";
    }
    if (base->run_task_baselines) {
        models[model_count++] = "single_task";
    }
    if (!mt_parse_seed_list(base->report_seeds, seeds, sizeof(seeds) / sizeof(seeds[0]), &seed_count)) {
        fputs("invalid --seeds list\n", stderr);
        return EXIT_FAILURE;
    }
    if (!mt_parse_capacity_list(base->tiny_net_capacities,
                                capacities,
                                sizeof(capacities) / sizeof(capacities[0]),
                                &capacity_count)) {
        fputs("invalid --capacities list\n", stderr);
        return EXIT_FAILURE;
    }
    if (!mt_parse_size_list(base->tiny_net_epoch_grid,
                            epoch_grid,
                            sizeof(epoch_grid) / sizeof(epoch_grid[0]),
                            &epoch_count)) {
        fputs("invalid --epoch-grid list\n", stderr);
        return EXIT_FAILURE;
    }
    {
        uint64_t signature = mt_tiny_audit_signature(base, epoch_grid, epoch_count);
        int written = snprintf(cell_dir_name,
                               sizeof(cell_dir_name),
                               "tiny_net_plateau_cells/%016llx",
                               (unsigned long long)signature);
        if (written < 0 || (size_t)written >= sizeof(cell_dir_name)) {
            fputs("failed to prepare tiny-net cell cache path\n", stderr);
            return EXIT_FAILURE;
        }
    }
    if (mt_mkdir_p(base->output_dir) != 0 ||
        mt_join_path(cell_dir_path, sizeof(cell_dir_path), base->output_dir, cell_dir_name) != 0 ||
        mt_mkdir_p(cell_dir_path) != 0 ||
        mt_join_path(raw_path, sizeof(raw_path), base->output_dir, "tiny_net_plateau.raw.jsonl") != 0 ||
        mt_join_path(commands_path, sizeof(commands_path), base->output_dir, "tiny_net_plateau.commands.txt") != 0 ||
        mt_join_path(curve_path, sizeof(curve_path), base->output_dir, "tiny_net_learning_curve.csv") != 0 ||
        mt_join_path(plateau_csv_path, sizeof(plateau_csv_path), base->output_dir, "tiny_net_plateau_summary.csv") != 0 ||
        mt_join_path(plateau_jsonl_path, sizeof(plateau_jsonl_path), base->output_dir, "tiny_net_plateau_summary.jsonl") != 0 ||
        mt_join_path(width_path, sizeof(width_path), base->output_dir, "tiny_net_width_help.jsonl") != 0 ||
        mt_join_path(verdict_path, sizeof(verdict_path), base->output_dir, "tiny_net_audit_verdict.jsonl") != 0 ||
        mt_join_path(diagnostics_path, sizeof(diagnostics_path), base->output_dir, "tiny_net_diagnostics.jsonl") != 0 ||
        mt_join_path(markdown_path, sizeof(markdown_path), base->output_dir, "tiny_net_plateau_audit.md") != 0) {
        fputs("failed to prepare tiny-net output paths\n", stderr);
        return EXIT_FAILURE;
    }
    commands = fopen(commands_path, "w");
    if (commands == NULL) {
        fputs("failed to open tiny-net command file\n", stderr);
        goto done;
    }
    fprintf(commands, "# tiny-net cell cache: %s\n", cell_dir_path);
    fprintf(commands,
            "# resume=%s shard=%zu/%zu\n",
            base->tiny_net_resume ? "true" : "false",
            base->tiny_net_shard_index,
            base->tiny_net_shard_count);
    for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
        for (size_t seed_index = 0U; seed_index < seed_count; ++seed_index) {
            MTConfig command_config = *base;
            char cell_path[4096];
            char tmp_cell_path[4096];
            char eval_task_set[128];
            size_t cell_index = cap_index * seed_count + seed_index;
            int selected = mt_tiny_shard_selected(cell_index,
                                                  base->tiny_net_shard_index,
                                                  base->tiny_net_shard_count);
            command_config.eval_task_mask =
                base->eval_task_mask == 0U ? all_task_mask : base->eval_task_mask;
            mt_task_set_name(command_config.eval_task_mask,
                             eval_task_set,
                             sizeof(eval_task_set));
            if (mt_tiny_cell_path(cell_path,
                                  sizeof(cell_path),
                                  cell_dir_path,
                                  &capacities[cap_index],
                                  seeds[seed_index]) != 0) {
                goto done;
            }
            fprintf(commands,
                    "multitask_domain --tiny-net-plateau-audit --output-dir \"%s\" --tasks all --eval-tasks %s --capacities %s --seeds %llu --epoch-grid %s --steps-per-epoch %zu --eval-samples %zu",
                    base->output_dir,
                    eval_task_set,
                    capacities[cap_index].label,
                    (unsigned long long)seeds[seed_index],
                    base->tiny_net_epoch_grid,
                    base->steps_per_epoch,
                    base->eval_samples);
            if (base->run_task_head_model) {
                fputs(" --task-heads", commands);
            } else {
                fputs(" --no-task-heads", commands);
            }
            if (!base->run_task_baselines) {
                fputs(" --no-task-baselines", commands);
            }
            fputc('\n', commands);
            if (selected) {
                int has_cell = mt_file_has_content(cell_path);
                if (base->tiny_net_resume && has_cell) {
                    printf("resumed tiny-net cell hidden=%s seed=%llu from %s\n",
                           capacities[cap_index].label,
                           (unsigned long long)seeds[seed_index],
                           cell_path);
                } else {
                    MTReportRows scratch_rows;
                    FILE *cell_raw;
                    int ok;
                    int written = snprintf(tmp_cell_path,
                                           sizeof(tmp_cell_path),
                                           "%s.tmp",
                                           cell_path);
                    if (written < 0 || (size_t)written >= sizeof(tmp_cell_path)) {
                        goto done;
                    }
                    memset(&scratch_rows, 0, sizeof(scratch_rows));
                    cell_raw = fopen(tmp_cell_path, "w");
                    if (cell_raw == NULL) {
                        free(scratch_rows.items);
                        goto done;
                    }
                    printf("running tiny-net cell hidden=%s seed=%llu -> %s\n",
                           capacities[cap_index].label,
                           (unsigned long long)seeds[seed_index],
                           cell_path);
                    ok = mt_run_tiny_net_plateau_cell(base,
                                                      &capacities[cap_index],
                                                      seeds[seed_index],
                                                      epoch_grid,
                                                      epoch_count,
                                                      &scratch_rows,
                                                      cell_raw);
                    if (fclose(cell_raw) != 0) {
                        ok = 0;
                    }
                    free(scratch_rows.items);
                    if (!ok) {
                        remove(tmp_cell_path);
                        goto done;
                    }
                    if (rename(tmp_cell_path, cell_path) != 0) {
                        remove(tmp_cell_path);
                        goto done;
                    }
                }
            }
            if (mt_file_has_content(cell_path)) {
                ++present_cell_count;
                if (mt_report_rows_load_jsonl(&rows, cell_path) != 0) {
                    fprintf(stderr, "failed to load tiny-net cell rows from %s\n", cell_path);
                    goto done;
                }
            } else {
                ++missing_cell_count;
            }
        }
    }
    fclose(commands);
    commands = NULL;
    raw = fopen(raw_path, "w");
    if (raw == NULL) {
        goto done;
    }
    for (size_t cap_index = 0U; cap_index < capacity_count; ++cap_index) {
        for (size_t seed_index = 0U; seed_index < seed_count; ++seed_index) {
            char cell_path[4096];
            if (mt_tiny_cell_path(cell_path,
                                  sizeof(cell_path),
                                  cell_dir_path,
                                  &capacities[cap_index],
                                  seeds[seed_index]) != 0) {
                goto done;
            }
            if (mt_file_has_content(cell_path) &&
                mt_copy_file_to(raw, cell_path) != 0) {
                goto done;
            }
        }
    }
    fclose(raw);
    raw = NULL;
    if (mt_tiny_collect_curves(&rows, &curves, &curve_count) != 0 ||
        mt_tiny_collect_plateaus(curves,
                                 curve_count,
                                 base->plateau_threshold,
                                 base->plateau_window,
                                 &plateaus,
                                 &plateau_count) != 0 ||
        mt_tiny_collect_widths(plateaus,
                               plateau_count,
                               capacities,
                               capacity_count,
                               domains,
                               domain_count,
                               models,
                               model_count,
                               &widths,
                               &width_count) != 0 ||
        mt_tiny_collect_verdicts(plateaus,
                                 plateau_count,
                                 widths,
                                 width_count,
                                 capacities,
                                 capacity_count,
                                 seeds,
                                 seed_count,
                                 epoch_grid,
                                 epoch_count,
                                 domains,
                                 domain_count,
                                 models,
                                 model_count,
                                 &verdicts,
                                 &verdict_count) != 0) {
        goto done;
    }
    if (missing_cell_count > 0U) {
        for (size_t i = 0U; i < verdict_count; ++i) {
            verdicts[i].capacity_coverage_complete = 0;
            verdicts[i].audit_ready = 0;
        }
        printf("tiny-net audit partial cell coverage: %zu/%zu cells present\n",
               present_cell_count,
               present_cell_count + missing_cell_count);
    }
    if (mt_write_tiny_learning_curve_csv(curve_path, curves, curve_count) != 0 ||
        mt_write_tiny_plateau_csv(plateau_csv_path, plateaus, plateau_count) != 0 ||
        mt_write_tiny_plateau_jsonl(plateau_jsonl_path, plateaus, plateau_count) != 0 ||
        mt_write_tiny_width_jsonl(width_path, widths, width_count, capacities, capacity_count) != 0 ||
        mt_write_tiny_verdict_jsonl(verdict_path,
                                    verdicts,
                                    verdict_count,
                                    capacities,
                                    capacity_count,
                                    epoch_grid,
                                    epoch_count,
                                    base->plateau_threshold,
                                    base->plateau_window) != 0 ||
        mt_write_tiny_diagnostics_jsonl(diagnostics_path,
                                        curves,
                                        curve_count,
                                        plateaus,
                                        plateau_count,
                                        widths,
                                        width_count,
                                        verdicts,
                                        verdict_count,
                                        capacities,
                                        capacity_count) != 0 ||
        mt_write_tiny_report_md(markdown_path,
                                seeds,
                                seed_count,
                                capacities,
                                capacity_count,
                                epoch_grid,
                                epoch_count,
                                base,
                                plateaus,
                                plateau_count,
                                widths,
                                width_count,
                                verdicts,
                                verdict_count,
                                domains,
                                domain_count,
                                models,
                                model_count) != 0) {
        goto done;
    }
    printf("wrote %s\n", markdown_path);
    printf("wrote %s\n", verdict_path);
    printf("wrote %s\n", diagnostics_path);
    status = EXIT_SUCCESS;
done:
    if (raw != NULL) {
        fclose(raw);
    }
    if (commands != NULL) {
        fclose(commands);
    }
    free(rows.items);
    free(curves);
    free(plateaus);
    free(widths);
    free(verdicts);
    return status;
}

static int mt_run_adapter_heads_report(const MTConfig *base)
{
    static const struct {
        const char *label;
        int suite;
    } suites[] = {
        {"fixed", MT_ARC_EVAL_0520FDE7},
        {"ood", MT_ARC_EVAL_OOD_SAME_ENVELOPE},
    };
    static const char *summary_domains[] = {"arc_agi", "delayed_claim", "grid_path", "prime_gap"};
    static const char *delta_domains[] = {"prime_gap", "grid_path", "delayed_claim", "arc_agi"};
    static const char *models[] = {"shared", "shared_task_heads", "single_task"};
    uint64_t seeds[64];
    size_t seed_count = 0U;
    MTReportRows rows;
    MTReportSummary summaries[32];
    size_t summary_count = 0U;
    MTReportDelta deltas[8];
    size_t delta_count = 0U;
    char raw_path[4096], commands_path[4096], summary_path[4096], delta_path[4096];
    char decision_path[4096], markdown_path[4096];
    FILE *raw = NULL;
    FILE *commands = NULL;
    int status = EXIT_FAILURE;
    memset(&rows, 0, sizeof(rows));
    if (!mt_parse_seed_list(base->report_seeds, seeds, sizeof(seeds) / sizeof(seeds[0]), &seed_count)) {
        fputs("invalid --seeds list\n", stderr);
        return EXIT_FAILURE;
    }
    if (mt_mkdir_p(base->output_dir) != 0 ||
        mt_join_path(raw_path, sizeof(raw_path), base->output_dir, "rmt4_adapter_heads.raw.jsonl") != 0 ||
        mt_join_path(commands_path, sizeof(commands_path), base->output_dir, "rmt4_adapter_heads.commands.txt") != 0 ||
        mt_join_path(summary_path, sizeof(summary_path), base->output_dir, "rmt4_adapter_heads_summary.csv") != 0 ||
        mt_join_path(delta_path, sizeof(delta_path), base->output_dir, "rmt4_adapter_heads_deltas.csv") != 0 ||
        mt_join_path(decision_path, sizeof(decision_path), base->output_dir, "rmt4_adapter_heads_decision.jsonl") != 0 ||
        mt_join_path(markdown_path, sizeof(markdown_path), base->output_dir, "rmt4_adapter_heads.md") != 0) {
        fputs("failed to prepare adapter-head output paths\n", stderr);
        return EXIT_FAILURE;
    }
    raw = fopen(raw_path, "w");
    commands = fopen(commands_path, "w");
    if (raw == NULL || commands == NULL) {
        fputs("failed to open adapter-head output files\n", stderr);
        goto done;
    }
    for (size_t seed_index = 0U; seed_index < seed_count; ++seed_index) {
        for (size_t suite_index = 0U; suite_index < sizeof(suites) / sizeof(suites[0]); ++suite_index) {
            MTConfig run = *base;
            run.adapter_heads_report = 0;
            run.seed = seeds[seed_index];
            run.arc_eval_suite = suites[suite_index].suite;
            run.task_mask = mt_task_bit(MT_TASK_PRIME_GAP) |
                            mt_task_bit(MT_TASK_GRID_PATH) |
                            mt_task_bit(MT_TASK_DELAYED_CLAIM) |
                            mt_task_bit(MT_TASK_ARC_AGI);
            run.eval_task_mask = run.task_mask;
            run.run_task_head_model = 1;
            run.quiet = 1;
            run.roadmap_item = "R-MT.4";
            run.adapter_eval_suite = suites[suite_index].label;
            run.json_output = raw;
            run.report_rows = &rows;
            fprintf(commands,
                    "multitask_domain --tasks all --eval-tasks all --epochs %zu --steps-per-epoch %zu --eval-samples %zu --hidden %zux%zu --seed %llu --arc-eval-suite %s --task-heads",
                    run.epochs,
                    run.steps_per_epoch,
                    run.eval_samples,
                    run.hidden1,
                    run.hidden2,
                    (unsigned long long)run.seed,
                    suites[suite_index].suite == MT_ARC_EVAL_OOD_SAME_ENVELOPE ? "ood" : "0520fde7");
            if (suites[suite_index].suite == MT_ARC_EVAL_OOD_SAME_ENVELOPE) {
                fputs(" --arc-ood-task ", commands);
                if (run.arc_ood_task < 0) {
                    fputs("all", commands);
                } else {
                    fprintf(commands, "%d", run.arc_ood_task);
                }
            }
            if (!run.run_task_baselines) {
                fputs(" --no-task-baselines", commands);
            }
            fputc('\n', commands);
            if (mt_run_once(&run) != EXIT_SUCCESS) {
                goto done;
            }
        }
    }
    fclose(raw);
    fclose(commands);
    raw = NULL;
    commands = NULL;
    for (size_t suite_index = 0U; suite_index < sizeof(suites) / sizeof(suites[0]); ++suite_index) {
        size_t domain_count =
            strcmp(suites[suite_index].label, "fixed") == 0 ? sizeof(summary_domains) / sizeof(summary_domains[0]) : 1U;
        for (size_t domain_index = 0U; domain_index < domain_count; ++domain_index) {
            const char *domain = strcmp(suites[suite_index].label, "fixed") == 0 ? summary_domains[domain_index] : "arc_agi";
            for (size_t model_index = 0U; model_index < sizeof(models) / sizeof(models[0]); ++model_index) {
                MTReportSummary summary;
                if (mt_summary_collect(&rows,
                                       suites[suite_index].label,
                                       domain,
                                       models[model_index],
                                       &summary)) {
                    summaries[summary_count++] = summary;
                }
            }
        }
    }
    for (size_t i = 0U; i < sizeof(delta_domains) / sizeof(delta_domains[0]); ++i) {
        MTReportDelta delta = mt_delta_build(summaries, summary_count, "fixed", delta_domains[i]);
        if (delta.present) {
            deltas[delta_count++] = delta;
        }
    }
    {
        MTReportDelta delta = mt_delta_build(summaries, summary_count, "ood", "arc_agi");
        if (delta.present) {
            deltas[delta_count++] = delta;
        }
    }
    if (mt_write_summary_csv(summary_path, summaries, summary_count) != 0 ||
        mt_write_deltas_csv(delta_path, deltas, delta_count) != 0 ||
        mt_write_decision_jsonl(decision_path,
                                deltas,
                                delta_count,
                                base->transfer_regression_margin) != 0 ||
        mt_write_adapter_report_md(markdown_path,
                                   seeds,
                                   seed_count,
                                   base,
                                   summaries,
                                   summary_count,
                                   deltas,
                                   delta_count) != 0) {
        goto done;
    }
    printf("wrote %s\n", markdown_path);
    status = EXIT_SUCCESS;
done:
    if (raw != NULL) {
        fclose(raw);
    }
    if (commands != NULL) {
        fclose(commands);
    }
    free(rows.items);
    return status;
}

int main(int argc, char **argv)
{
    MTConfig config;
    if (!mt_parse_config(argc, argv, &config)) {
        mt_print_usage(stderr);
        return EXIT_FAILURE;
    }
    if (config.adapter_heads_report) {
        return mt_run_adapter_heads_report(&config);
    }
    if (config.tiny_net_plateau_audit) {
        return mt_run_tiny_net_plateau_audit(&config);
    }
    return mt_run_once(&config);
}
