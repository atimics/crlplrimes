#include "nn.h"
#include "nn_experiment.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SF_TRACE_SCHEMA "signal.flight_trace.v1"
#define SF_COUNTERFACTUAL_TRACE_SCHEMA "signal.flight_counterfactual.v1"
#define SF_SUMMARY_SCHEMA "crlp.signal_flight.summary.v1"
#define SF_ACTION_COUNT 9u
#define SF_FEATURE_COUNT 48u
#define SF_CALIBRATION_BINS 10u
#define SF_MIN_TRACE_FIELDS 34u
#define SF_LINE_SIZE 4096u

typedef struct {
    int action;
    uint32_t candidate_mask;
    uint64_t group_id;
    int has_utility;
    double teacher_turn;
    double teacher_thrust;
    double dist;
    double control_dist;
    double heading_cos;
    double heading_sin;
    double control_heading_cos;
    double control_heading_sin;
    double speed;
    double forward_speed;
    double lateral_speed;
    double brake_distance;
    double fwd_clear;
    double left_clear;
    double right_clear;
    double vel_clear;
    double signal;
    double hull_ratio;
    double path_count;
    double path_current;
    double progress;
    double next_dist;
    double next_speed;
    double utility;
    double damage_amount;
    double hull_loss;
    int damage_events;
    int death_events;
} SFTraceRow;

typedef struct {
    SFTraceRow *items;
    size_t count;
    size_t capacity;
    int has_utility;
} SFDataset;

typedef struct {
    size_t start;
    size_t count;
} SFUtilityGroup;

typedef struct {
    const char *train_path;
    const char *test_path;
    const char *save_path;
    const char *summary_path;
    size_t epochs;
    size_t max_train;
    size_t max_test;
    uint64_t seed;
    nn_real learning_rate;
    int live_features;
} SFConfig;

typedef struct {
    size_t samples;
    size_t top1;
    size_t top3;
    size_t selected_in_mask;
    size_t teacher_in_mask;
    size_t target_counts[SF_ACTION_COUNT];
    size_t selected_counts[SF_ACTION_COUNT];
    size_t confusion[SF_ACTION_COUNT][SF_ACTION_COUNT];
    size_t bin_count[SF_CALIBRATION_BINS];
    double bin_confidence[SF_CALIBRATION_BINS];
    double bin_accuracy[SF_CALIBRATION_BINS];
    double nll;
    double mean_rank;
    double ece;
    double confidence;
    size_t utility_groups;
    size_t selected_damage;
    size_t oracle_damage;
    size_t selected_death;
    size_t oracle_death;
    size_t baseline_samples;
    size_t baseline_top1;
    size_t baseline_damage;
    size_t baseline_death;
    double selected_utility;
    double oracle_utility;
    double regret;
    double baseline_selected_utility;
    double baseline_oracle_utility;
    double baseline_regret;
} SFMetrics;

static const char *SF_ACTION_NAMES[SF_ACTION_COUNT] = {
    "NONE", "W", "A", "D", "S", "WA", "WD", "SA", "SD",
};

static int g_sf_live_features = 0;

static const NNFeatureSchema SF_FEATURES[SF_FEATURE_COUNT] = {
    {"bias", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_action_index", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_turn", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_thrust", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_turn_left", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_turn_right", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_forward", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_brake", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_idle", NN_FEATURE_STATE_OBSERVATION, 1},
    {"goal_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"control_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"goal_heading_cos", NN_FEATURE_STATE_OBSERVATION, 1},
    {"goal_heading_sin", NN_FEATURE_STATE_OBSERVATION, 1},
    {"control_heading_cos", NN_FEATURE_STATE_OBSERVATION, 1},
    {"control_heading_sin", NN_FEATURE_STATE_OBSERVATION, 1},
    {"speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"forward_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"lateral_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"brake_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"forward_clearance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"left_clearance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"right_clearance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"velocity_clearance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"signal_quality", NN_FEATURE_STATE_OBSERVATION, 1},
    {"hull_ratio", NN_FEATURE_STATE_OBSERVATION, 1},
    {"path_count", NN_FEATURE_STATE_OBSERVATION, 1},
    {"path_current", NN_FEATURE_STATE_OBSERVATION, 1},
    {"step_progress", NN_FEATURE_STATE_OBSERVATION, 1},
    {"next_goal_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"next_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"turn_by_control_heading", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_control_heading", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_forward_clearance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"brake_by_brake_distance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"left_turn_by_left_clearance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"right_turn_by_right_clearance", NN_FEATURE_STATE_OBSERVATION, 1},
    {"idle_by_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"turn_by_lateral_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_forward_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_hull_margin", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_next_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_progress", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_close_goal", NN_FEATURE_STATE_OBSERVATION, 1},
    {"turn_by_goal_heading", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_by_goal_heading", NN_FEATURE_STATE_OBSERVATION, 1},
    {"brake_by_forward_speed", NN_FEATURE_STATE_OBSERVATION, 1},
    {"turn_abs", NN_FEATURE_STATE_OBSERVATION, 1},
    {"thrust_abs", NN_FEATURE_STATE_OBSERVATION, 1},
};

static const NNExperimentSpec SF_EXPERIMENT_SPEC = {
    .task_name = "signal_flight",
    .role = "Signal headless simulator trace reduced to WASD action scoring",
    .state_space = "ship pose, velocity, hull, path, local clearance sensors, and target/control-point geometry",
    .state_distribution = "episodes generated by ../signal/build/flight_trace from deterministic simulator seeds",
    .action_space = "nine discrete player-input actions: NONE, W, A, D, S, WA, WD, SA, SD",
    .candidate_enumerator = "enumerate every WASD key combination supported by the first trace-brain interface",
    .observable_features = "candidate key controls plus normalized ship kinematics, heading, path, hull, and clearance observations",
    .leakage_policy = NN_LEAKAGE_FORBIDDEN,
    .neural_scorer = "tiny MLP scores one state/action row and softmax ranks WASD candidates",
    .exact_verifier = "deterministic Signal authoritative replay for each WASD hypothesis; legacy teacher traces remain accepted for ablation",
    .verifier_feedback = "certified utility, straight-forward progress, wall-aware clearance, hull loss, damage/death events, held-out top-k regret, NLL, calibration, and candidate-mask sanity rate",
    .rule_grammar = "WASD control hypotheses now; planned higher-level clearance, speed, braking-distance, and hull-threshold predicates",
    .certificate_checker = "deterministic simulator replay that records collision pain instead of suppressing damage in prediction mode, including station-wall probes",
    .gate_executor = "not active; current experiment reports raw neural scorer without a hand safety wrapper",
    .proposer_inputs = "observable state/action rows only; no candidate-mask bit or teacher action is a scorer feature",
    .proposer_targets = "argmax over certified counterfactual utility rows that reward nose-forward progress; deterministic teacher imitation remains an explicit legacy ablation",
    .proposer_mode = NN_PROPOSER_SELECTION,
    .baseline_orderings = "goal-heading no-safety WASD baseline, uniform random, teacher trace action distribution, and raw neural scorer",
    .generalization_axis = "train simulator seeds to held-out simulator seeds with identical ship-control interface",
    .trace_schema = SF_TRACE_SCHEMA "," SF_COUNTERFACTUAL_TRACE_SCHEMA,
    .replay_protocol = "CSV trace replay with stable action ids matching Signal brain.c WASD ordering",
    .rl_protocol = "counterfactual replay from deterministic simulator states; future closed-loop rollouts can append the same pain-certified rows",
    .features = SF_FEATURES,
    .feature_count = sizeof(SF_FEATURES) / sizeof(SF_FEATURES[0]),
};

static double sf_clip(double value, double min_value, double max_value)
{
    if (!isfinite(value)) {
        return 0.0;
    }
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static double sf_unit(double value)
{
    return sf_clip(value, 0.0, 1.0);
}

static double sf_scale(double value, double scale)
{
    if (scale <= 0.0) {
        return 0.0;
    }
    return sf_clip(value / scale, 0.0, 1.0);
}

static double sf_signed_scale(double value, double scale)
{
    if (scale <= 0.0) {
        return 0.0;
    }
    return sf_clip(value / scale, -1.0, 1.0);
}

static double sf_safe_probability(double probability)
{
    if (probability < 1e-12) {
        return 1e-12;
    }
    if (probability > 1.0) {
        return 1.0;
    }
    return probability;
}

static uint64_t sf_rng_next(uint64_t *state)
{
    *state = (*state * UINT64_C(6364136223846793005)) + UINT64_C(1442695040888963407);
    return *state;
}

static int sf_parse_size(const char *text, size_t *out_value)
{
    if (text == NULL || text[0] == '\0' || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out_value = (size_t)value;
    return 1;
}

static int sf_parse_u64(const char *text, uint64_t *out_value)
{
    if (text == NULL || text[0] == '\0' || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return 0;
    }
    *out_value = (uint64_t)value;
    return 1;
}

static int sf_parse_real(const char *text, nn_real *out_value)
{
    if (text == NULL || text[0] == '\0' || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(value) || value <= 0.0) {
        return 0;
    }
    *out_value = (nn_real)value;
    return 1;
}

static int sf_parse_double_field(const char *text, double *out_value)
{
    if (text == NULL || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (errno != 0 || end == text || (*end != '\0' && *end != '\n' && *end != '\r')) {
        return 0;
    }
    *out_value = value;
    return 1;
}

static int sf_parse_int_field(const char *text, int *out_value)
{
    if (text == NULL || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    long value = strtol(text, &end, 10);
    if (errno != 0 || end == text || (*end != '\0' && *end != '\n' && *end != '\r')) {
        return 0;
    }
    *out_value = (int)value;
    return 1;
}

static int sf_parse_u32_field(const char *text, uint32_t *out_value)
{
    if (text == NULL || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || (*end != '\0' && *end != '\n' && *end != '\r')) {
        return 0;
    }
    *out_value = (uint32_t)value;
    return 1;
}

static int sf_parse_u64_field(const char *text, uint64_t *out_value)
{
    if (text == NULL || out_value == NULL) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || (*end != '\0' && *end != '\n' && *end != '\r')) {
        return 0;
    }
    *out_value = (uint64_t)value;
    return 1;
}

static size_t sf_split_csv(char *line, char **fields, size_t capacity)
{
    size_t count = 0;
    char *cursor = line;
    while (cursor != NULL && *cursor != '\0') {
        if (count < capacity) {
            fields[count] = cursor;
        }
        ++count;
        while (*cursor != '\0' && *cursor != ',' && *cursor != '\n' && *cursor != '\r') {
            ++cursor;
        }
        if (*cursor == ',') {
            *cursor = '\0';
            ++cursor;
            continue;
        }
        if (*cursor != '\0') {
            *cursor = '\0';
        }
        break;
    }
    return count;
}

static void sf_dataset_free(SFDataset *dataset)
{
    if (dataset != NULL) {
        free(dataset->items);
        dataset->items = NULL;
        dataset->count = 0;
        dataset->capacity = 0;
    }
}

static int sf_dataset_push(SFDataset *dataset, const SFTraceRow *row)
{
    if (dataset == NULL || row == NULL) {
        return 0;
    }
    if (dataset->count == dataset->capacity) {
        size_t new_capacity = dataset->capacity == 0 ? 1024u : dataset->capacity * 2u;
        SFTraceRow *new_items = realloc(dataset->items, new_capacity * sizeof(*new_items));
        if (new_items == NULL) {
            return 0;
        }
        dataset->items = new_items;
        dataset->capacity = new_capacity;
    }
    dataset->items[dataset->count++] = *row;
    return 1;
}

static int sf_parse_trace_fields(char **fields, size_t field_count, SFTraceRow *row)
{
    int is_counterfactual = 0;
    if (fields == NULL || row == NULL) {
        return 0;
    }
    if (strcmp(fields[0], SF_TRACE_SCHEMA) == 0) {
        is_counterfactual = 0;
    } else if (strcmp(fields[0], SF_COUNTERFACTUAL_TRACE_SCHEMA) == 0) {
        is_counterfactual = 1;
    } else {
        return 0;
    }
    memset(row, 0, sizeof(*row));
    if (!sf_parse_int_field(fields[6], &row->action) ||
        !sf_parse_u32_field(fields[10], &row->candidate_mask) ||
        !sf_parse_double_field(fields[11], &row->teacher_turn) ||
        !sf_parse_double_field(fields[12], &row->teacher_thrust) ||
        !sf_parse_double_field(fields[13], &row->dist) ||
        !sf_parse_double_field(fields[14], &row->control_dist) ||
        !sf_parse_double_field(fields[15], &row->heading_cos) ||
        !sf_parse_double_field(fields[16], &row->heading_sin) ||
        !sf_parse_double_field(fields[17], &row->control_heading_cos) ||
        !sf_parse_double_field(fields[18], &row->control_heading_sin) ||
        !sf_parse_double_field(fields[19], &row->speed) ||
        !sf_parse_double_field(fields[20], &row->forward_speed) ||
        !sf_parse_double_field(fields[21], &row->lateral_speed) ||
        !sf_parse_double_field(fields[22], &row->brake_distance) ||
        !sf_parse_double_field(fields[23], &row->fwd_clear) ||
        !sf_parse_double_field(fields[24], &row->left_clear) ||
        !sf_parse_double_field(fields[25], &row->right_clear) ||
        !sf_parse_double_field(fields[26], &row->vel_clear) ||
        !sf_parse_double_field(fields[27], &row->signal) ||
        !sf_parse_double_field(fields[28], &row->hull_ratio) ||
        !sf_parse_double_field(fields[29], &row->path_count) ||
        !sf_parse_double_field(fields[30], &row->path_current) ||
        !sf_parse_double_field(fields[31], &row->progress) ||
        !sf_parse_double_field(fields[32], &row->next_dist) ||
        !sf_parse_double_field(fields[33], &row->next_speed)) {
        return 0;
    }
    if (is_counterfactual) {
        if (field_count < 51u ||
            !sf_parse_u64_field(fields[43], &row->group_id) ||
            !sf_parse_double_field(fields[45], &row->utility) ||
            !sf_parse_int_field(fields[46], &row->damage_events) ||
            !sf_parse_int_field(fields[47], &row->death_events) ||
            !sf_parse_double_field(fields[48], &row->damage_amount) ||
            !sf_parse_double_field(fields[49], &row->hull_loss)) {
            return 0;
        }
        row->has_utility = 1;
    }
    return row->action >= 0 && row->action < (int)SF_ACTION_COUNT;
}

static int sf_load_trace_csv(const char *path, size_t max_rows, SFDataset *dataset)
{
    if (path == NULL || dataset == NULL) {
        return 0;
    }
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open Signal trace: %s\n", path);
        return 0;
    }

    char line[SF_LINE_SIZE];
    size_t line_number = 0;
    int ok = 1;
    while (fgets(line, sizeof(line), fp) != NULL) {
        ++line_number;
        if (line[0] == '\n' || line[0] == '\r' || line[0] == '\0') {
            continue;
        }

        char *fields[64] = {0};
        size_t field_count = sf_split_csv(line, fields, sizeof(fields) / sizeof(fields[0]));
        if (field_count == 0) {
            continue;
        }
        if (strcmp(fields[0], "schema") == 0) {
            continue;
        }
        if (field_count < SF_MIN_TRACE_FIELDS) {
            fprintf(stderr, "%s:%zu: expected at least %u fields, saw %zu\n",
                    path,
                    line_number,
                    (unsigned)SF_MIN_TRACE_FIELDS,
                    field_count);
            ok = 0;
            break;
        }

        SFTraceRow row;
        if (!sf_parse_trace_fields(fields, field_count, &row)) {
            fprintf(stderr, "%s:%zu: invalid Signal flight trace row\n", path, line_number);
            ok = 0;
            break;
        }
        if (row.has_utility) {
            dataset->has_utility = 1;
        }
        if (!sf_dataset_push(dataset, &row)) {
            fprintf(stderr, "out of memory while loading %s\n", path);
            ok = 0;
            break;
        }
        if (max_rows != 0 && dataset->count >= max_rows) {
            break;
        }
    }

    fclose(fp);
    if (ok && dataset->count == 0) {
        fprintf(stderr, "Signal trace has no usable rows: %s\n", path);
        ok = 0;
    }
    return ok;
}

static void sf_action_controls(size_t action, double *turn, double *thrust)
{
    double t = 0.0;
    double f = 0.0;
    switch (action) {
    case 1:
        f = 1.0;
        break;
    case 2:
        t = -1.0;
        break;
    case 3:
        t = 1.0;
        break;
    case 4:
        f = -1.0;
        break;
    case 5:
        t = -1.0;
        f = 1.0;
        break;
    case 6:
        t = 1.0;
        f = 1.0;
        break;
    case 7:
        t = -1.0;
        f = -1.0;
        break;
    case 8:
        t = 1.0;
        f = -1.0;
        break;
    default:
        break;
    }
    if (turn != NULL) {
        *turn = t;
    }
    if (thrust != NULL) {
        *thrust = f;
    }
}

static size_t sf_action_from_controls(int turn, int thrust)
{
    if (turn == 0 && thrust == 1) {
        return 1u;
    }
    if (turn == -1 && thrust == 0) {
        return 2u;
    }
    if (turn == 1 && thrust == 0) {
        return 3u;
    }
    if (turn == 0 && thrust == -1) {
        return 4u;
    }
    if (turn == -1 && thrust == 1) {
        return 5u;
    }
    if (turn == 1 && thrust == 1) {
        return 6u;
    }
    if (turn == -1 && thrust == -1) {
        return 7u;
    }
    if (turn == 1 && thrust == -1) {
        return 8u;
    }
    return 0u;
}

static size_t sf_goal_heading_baseline_action(const SFTraceRow *trace)
{
    int turn = 0;
    int thrust = 0;
    if (trace == NULL) {
        return 0u;
    }
    if (trace->heading_sin < -0.18) {
        turn = -1;
    } else if (trace->heading_sin > 0.18) {
        turn = 1;
    }
    if (trace->heading_cos > 0.73) {
        thrust = 1;
    }
    return sf_action_from_controls(turn, thrust);
}

static void sf_fill_action_row(const SFTraceRow *trace, size_t action, nn_real *row)
{
    double turn = 0.0;
    double thrust = 0.0;
    sf_action_controls(action, &turn, &thrust);

    double dist = sf_scale(trace->dist, 6000.0);
    double control_dist = sf_scale(trace->control_dist, 6000.0);
    double speed = sf_signed_scale(trace->speed, 350.0);
    double forward_speed = sf_signed_scale(trace->forward_speed, 350.0);
    double lateral_speed = sf_signed_scale(trace->lateral_speed, 350.0);
    double brake_distance = sf_scale(trace->brake_distance, 700.0);
    double fwd_clear = sf_unit(trace->fwd_clear);
    double left_clear = sf_unit(trace->left_clear);
    double right_clear = sf_unit(trace->right_clear);
    double vel_clear = sf_unit(trace->vel_clear);
    double signal = sf_unit(trace->signal);
    double hull = sf_unit(trace->hull_ratio);
    double path_count = sf_scale(trace->path_count, 16.0);
    double path_current = sf_scale(trace->path_current, 16.0);
    double progress = sf_signed_scale(trace->progress, 350.0);
    double next_dist = sf_scale(trace->next_dist, 6000.0);
    double next_speed = sf_signed_scale(trace->next_speed, 350.0);
    double close_goal = 1.0 - dist;

    if (g_sf_live_features) {
        progress = 0.0;
        next_dist = dist;
        next_speed = speed;
    }

    memset(row, 0, SF_FEATURE_COUNT * sizeof(*row));
    row[0] = (nn_real)1;
    row[1] = (nn_real)((double)action / (double)(SF_ACTION_COUNT - 1u));
    row[2] = (nn_real)turn;
    row[3] = (nn_real)thrust;
    row[4] = (nn_real)(turn < 0.0 ? 1.0 : 0.0);
    row[5] = (nn_real)(turn > 0.0 ? 1.0 : 0.0);
    row[6] = (nn_real)(thrust > 0.0 ? 1.0 : 0.0);
    row[7] = (nn_real)(thrust < 0.0 ? 1.0 : 0.0);
    row[8] = (nn_real)(action == 0 ? 1.0 : 0.0);
    row[9] = (nn_real)dist;
    row[10] = (nn_real)control_dist;
    row[11] = (nn_real)sf_clip(trace->heading_cos, -1.0, 1.0);
    row[12] = (nn_real)sf_clip(trace->heading_sin, -1.0, 1.0);
    row[13] = (nn_real)sf_clip(trace->control_heading_cos, -1.0, 1.0);
    row[14] = (nn_real)sf_clip(trace->control_heading_sin, -1.0, 1.0);
    row[15] = (nn_real)speed;
    row[16] = (nn_real)forward_speed;
    row[17] = (nn_real)lateral_speed;
    row[18] = (nn_real)brake_distance;
    row[19] = (nn_real)fwd_clear;
    row[20] = (nn_real)left_clear;
    row[21] = (nn_real)right_clear;
    row[22] = (nn_real)vel_clear;
    row[23] = (nn_real)signal;
    row[24] = (nn_real)hull;
    row[25] = (nn_real)path_count;
    row[26] = (nn_real)path_current;
    row[27] = (nn_real)progress;
    row[28] = (nn_real)next_dist;
    row[29] = (nn_real)next_speed;
    row[30] = (nn_real)(turn * sf_clip(trace->control_heading_sin, -1.0, 1.0));
    row[31] = (nn_real)(thrust * sf_clip(trace->control_heading_cos, -1.0, 1.0));
    row[32] = (nn_real)(thrust * fwd_clear);
    row[33] = (nn_real)((thrust < 0.0 ? 1.0 : 0.0) * brake_distance);
    row[34] = (nn_real)((turn < 0.0 ? 1.0 : 0.0) * left_clear);
    row[35] = (nn_real)((turn > 0.0 ? 1.0 : 0.0) * right_clear);
    row[36] = (nn_real)((action == 0 ? 1.0 : 0.0) * fabs(speed));
    row[37] = (nn_real)(turn * lateral_speed);
    row[38] = (nn_real)(thrust * forward_speed);
    row[39] = (nn_real)(thrust * (1.0 - hull));
    row[40] = (nn_real)(thrust * next_speed);
    row[41] = (nn_real)(thrust * progress);
    row[42] = (nn_real)(thrust * close_goal);
    row[43] = (nn_real)(turn * sf_clip(trace->heading_sin, -1.0, 1.0));
    row[44] = (nn_real)(thrust * sf_clip(trace->heading_cos, -1.0, 1.0));
    row[45] = (nn_real)((thrust < 0.0 ? 1.0 : 0.0) * forward_speed);
    row[46] = (nn_real)fabs(turn);
    row[47] = (nn_real)fabs(thrust);
}

static size_t sf_fill_candidates(const SFTraceRow *trace, nn_real *candidate_inputs)
{
    for (size_t action = 0; action < SF_ACTION_COUNT; ++action) {
        sf_fill_action_row(trace, action, &candidate_inputs[action * SF_FEATURE_COUNT]);
    }
    return SF_ACTION_COUNT;
}

static SFUtilityGroup *sf_collect_utility_groups(const SFDataset *dataset,
                                                 size_t *out_count)
{
    SFUtilityGroup *groups = NULL;
    size_t count = 0;
    size_t capacity = 0;

    if (out_count != NULL) {
        *out_count = 0;
    }
    if (dataset == NULL || out_count == NULL || !dataset->has_utility) {
        return NULL;
    }

    for (size_t i = 0; i < dataset->count;) {
        uint64_t group_id = dataset->items[i].group_id;
        size_t start = i;
        while (i < dataset->count && dataset->items[i].group_id == group_id) {
            ++i;
        }
        if (i - start < 2u) {
            continue;
        }
        if (count == capacity) {
            size_t new_capacity = capacity == 0u ? 256u : capacity * 2u;
            SFUtilityGroup *new_groups =
                realloc(groups, new_capacity * sizeof(*new_groups));
            if (new_groups == NULL) {
                free(groups);
                return NULL;
            }
            groups = new_groups;
            capacity = new_capacity;
        }
        groups[count++] = (SFUtilityGroup){.start = start, .count = i - start};
    }

    *out_count = count;
    return groups;
}

static int sf_group_values(const SFDataset *dataset,
                           const SFUtilityGroup *group,
                           double utilities[SF_ACTION_COUNT],
                           int present[SF_ACTION_COUNT],
                           int damage_events[SF_ACTION_COUNT],
                           int death_events[SF_ACTION_COUNT],
                           size_t *out_best,
                           double *out_second_best)
{
    size_t best = 0u;
    double best_value = -HUGE_VAL;
    double second_best = -HUGE_VAL;

    if (dataset == NULL || group == NULL || utilities == NULL || present == NULL ||
        damage_events == NULL || death_events == NULL || out_best == NULL ||
        out_second_best == NULL) {
        return 0;
    }

    for (size_t i = 0; i < SF_ACTION_COUNT; ++i) {
        utilities[i] = -HUGE_VAL;
        present[i] = 0;
        damage_events[i] = 0;
        death_events[i] = 0;
    }

    for (size_t i = 0; i < group->count; ++i) {
        const SFTraceRow *row = &dataset->items[group->start + i];
        if (row->action < 0 || row->action >= (int)SF_ACTION_COUNT) {
            continue;
        }
        size_t action = (size_t)row->action;
        utilities[action] = row->utility;
        present[action] = 1;
        damage_events[action] = row->damage_events;
        death_events[action] = row->death_events;
    }

    for (size_t action = 0; action < SF_ACTION_COUNT; ++action) {
        if (!present[action]) {
            continue;
        }
        if (utilities[action] > best_value) {
            second_best = best_value;
            best_value = utilities[action];
            best = action;
        } else if (utilities[action] > second_best) {
            second_best = utilities[action];
        }
    }

    if (!isfinite(best_value)) {
        return 0;
    }
    if (!isfinite(second_best)) {
        second_best = best_value;
    }
    *out_best = best;
    *out_second_best = second_best;
    return 1;
}

static NeuralNet *sf_create_model(unsigned int seed)
{
    static const size_t layers[] = {SF_FEATURE_COUNT, 32u, 16u, 1u};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int sf_train_utility_model(NeuralNet *net, const SFDataset *train, const SFConfig *config)
{
    if (net == NULL || train == NULL || config == NULL || train->count == 0) {
        return 0;
    }

    size_t group_count = 0u;
    SFUtilityGroup *groups = sf_collect_utility_groups(train, &group_count);
    if (groups == NULL || group_count == 0u) {
        free(groups);
        return 0;
    }

    nn_real *candidate_inputs =
        calloc(SF_ACTION_COUNT * SF_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        free(groups);
        return 0;
    }

    uint64_t rng = config->seed ^ (config->seed >> 32) ^ UINT64_C(0x5eedcafe);
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double loss_sum = 0.0;
        for (size_t step = 0; step < group_count; ++step) {
            size_t group_index = (size_t)(sf_rng_next(&rng) % (uint64_t)group_count);
            const SFUtilityGroup *group = &groups[group_index];
            const SFTraceRow *trace = &train->items[group->start];
            double utilities[SF_ACTION_COUNT];
            int present[SF_ACTION_COUNT];
            int damage_events[SF_ACTION_COUNT];
            int death_events[SF_ACTION_COUNT];
            size_t target = 0u;
            double second_best = 0.0;
            if (!sf_group_values(train,
                                 group,
                                 utilities,
                                 present,
                                 damage_events,
                                 death_events,
                                 &target,
                                 &second_best)) {
                continue;
            }
            size_t candidate_count = sf_fill_candidates(trace, candidate_inputs);
            double gap = utilities[target] - second_best;
            nn_real weight = (nn_real)(1.0 + sf_clip(gap, 0.0, 5.0));
            nn_real loss = nn_train_softmax_scorer_weighted(net,
                                                            candidate_inputs,
                                                            candidate_count,
                                                            target,
                                                            weight,
                                                            config->learning_rate);
            if (!isfinite((double)loss)) {
                free(candidate_inputs);
                free(groups);
                return 0;
            }
            loss_sum += (double)loss;
        }
        printf("signal_flight counterfactual epoch %zu loss=%.4f groups=%zu\n",
               epoch + 1u,
               loss_sum / (double)group_count,
               group_count);
    }

    free(candidate_inputs);
    free(groups);
    return 1;
}

static int sf_train_model(NeuralNet *net, const SFDataset *train, const SFConfig *config)
{
    if (net == NULL || train == NULL || config == NULL || train->count == 0) {
        return 0;
    }
    if (train->has_utility) {
        return sf_train_utility_model(net, train, config);
    }

    nn_real *candidate_inputs =
        calloc(SF_ACTION_COUNT * SF_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        return 0;
    }

    uint64_t rng = config->seed ^ (config->seed >> 32) ^ UINT64_C(0x5eed1234);
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double loss_sum = 0.0;
        for (size_t step = 0; step < train->count; ++step) {
            size_t index = (size_t)(sf_rng_next(&rng) % (uint64_t)train->count);
            const SFTraceRow *trace = &train->items[index];
            size_t candidate_count = sf_fill_candidates(trace, candidate_inputs);
            nn_real loss = nn_train_action_scorer(net,
                                                  candidate_inputs,
                                                  candidate_count,
                                                  (size_t)trace->action,
                                                  config->learning_rate);
            if (!isfinite((double)loss)) {
                free(candidate_inputs);
                return 0;
            }
            loss_sum += (double)loss;
        }
        printf("signal_flight epoch %zu loss=%.4f samples=%zu\n",
               epoch + 1u,
               loss_sum / (double)train->count,
               train->count);
    }

    free(candidate_inputs);
    return 1;
}

static SFMetrics sf_evaluate_utility_model(NeuralNet *net, const SFDataset *test)
{
    SFMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    size_t group_count = 0u;
    SFUtilityGroup *groups = sf_collect_utility_groups(test, &group_count);
    nn_real *candidate_inputs =
        calloc(SF_ACTION_COUNT * SF_FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(SF_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(SF_ACTION_COUNT, sizeof(*probabilities));
    size_t *ranked = calloc(SF_ACTION_COUNT, sizeof(*ranked));
    if (groups == NULL || group_count == 0u || candidate_inputs == NULL ||
        scores == NULL || probabilities == NULL || ranked == NULL) {
        free(groups);
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        free(ranked);
        return metrics;
    }

    for (size_t i = 0; i < group_count; ++i) {
        const SFUtilityGroup *group = &groups[i];
        const SFTraceRow *trace = &test->items[group->start];
        double utilities[SF_ACTION_COUNT];
        int present[SF_ACTION_COUNT];
        int damage_events[SF_ACTION_COUNT];
        int death_events[SF_ACTION_COUNT];
        size_t target = 0u;
        double second_best = 0.0;
        if (!sf_group_values(test,
                             group,
                             utilities,
                             present,
                             damage_events,
                             death_events,
                             &target,
                             &second_best)) {
            continue;
        }

        size_t candidate_count = sf_fill_candidates(trace, candidate_inputs);
        if (!nn_score_action_rows(net, candidate_inputs, candidate_count, scores, probabilities) ||
            !nn_rank_action_scores(scores, candidate_count, ranked)) {
            continue;
        }

        size_t selected = ranked[0];
        size_t rank = candidate_count;
        for (size_t j = 0; j < candidate_count; ++j) {
            if (ranked[j] == target) {
                rank = j;
                break;
            }
        }
        if (rank >= candidate_count || !present[selected]) {
            continue;
        }

        size_t baseline = sf_goal_heading_baseline_action(trace);
        if (baseline >= SF_ACTION_COUNT || !present[baseline]) {
            baseline = selected;
            for (size_t action = 0; action < SF_ACTION_COUNT; ++action) {
                if (present[action]) {
                    baseline = action;
                    break;
                }
            }
        }

        double confidence = (double)probabilities[selected];
        size_t bin = confidence >= 1.0 ? SF_CALIBRATION_BINS - 1u
                                       : (size_t)(confidence * (double)SF_CALIBRATION_BINS);
        if (bin >= SF_CALIBRATION_BINS) {
            bin = SF_CALIBRATION_BINS - 1u;
        }

        ++metrics.samples;
        ++metrics.utility_groups;
        metrics.top1 += rank == 0 ? 1u : 0u;
        metrics.top3 += rank < 3u ? 1u : 0u;
        metrics.selected_in_mask +=
            (trace->candidate_mask & (UINT32_C(1) << selected)) != 0u ? 1u : 0u;
        metrics.teacher_in_mask +=
            (trace->candidate_mask & (UINT32_C(1) << target)) != 0u ? 1u : 0u;
        ++metrics.target_counts[target];
        ++metrics.selected_counts[selected];
        ++metrics.confusion[target][selected];
        metrics.nll += -log(sf_safe_probability((double)probabilities[target]));
        metrics.mean_rank += (double)rank + 1.0;
        metrics.confidence += confidence;
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += rank == 0 ? 1.0 : 0.0;
        metrics.selected_utility += utilities[selected];
        metrics.oracle_utility += utilities[target];
        metrics.regret += utilities[target] - utilities[selected];
        metrics.selected_damage += damage_events[selected] > 0 ? 1u : 0u;
        metrics.oracle_damage += damage_events[target] > 0 ? 1u : 0u;
        metrics.selected_death += death_events[selected] > 0 ? 1u : 0u;
        metrics.oracle_death += death_events[target] > 0 ? 1u : 0u;
        ++metrics.baseline_samples;
        metrics.baseline_top1 += baseline == target ? 1u : 0u;
        metrics.baseline_selected_utility += utilities[baseline];
        metrics.baseline_oracle_utility += utilities[target];
        metrics.baseline_regret += utilities[target] - utilities[baseline];
        metrics.baseline_damage += damage_events[baseline] > 0 ? 1u : 0u;
        metrics.baseline_death += death_events[baseline] > 0 ? 1u : 0u;
        (void)second_best;
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    double baseline_count =
        metrics.baseline_samples == 0 ? 1.0 : (double)metrics.baseline_samples;
    metrics.nll /= sample_count;
    metrics.mean_rank /= sample_count;
    metrics.confidence /= sample_count;
    metrics.selected_utility /= sample_count;
    metrics.oracle_utility /= sample_count;
    metrics.regret /= sample_count;
    metrics.baseline_selected_utility /= baseline_count;
    metrics.baseline_oracle_utility /= baseline_count;
    metrics.baseline_regret /= baseline_count;
    for (size_t bin = 0; bin < SF_CALIBRATION_BINS; ++bin) {
        if (metrics.bin_count[bin] == 0) {
            continue;
        }
        metrics.bin_confidence[bin] /= (double)metrics.bin_count[bin];
        metrics.bin_accuracy[bin] /= (double)metrics.bin_count[bin];
        metrics.ece += fabs(metrics.bin_confidence[bin] - metrics.bin_accuracy[bin]) *
                       ((double)metrics.bin_count[bin] / sample_count);
    }

    free(groups);
    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free(ranked);
    return metrics;
}

static SFMetrics sf_evaluate_model(NeuralNet *net, const SFDataset *test)
{
    SFMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));

    if (test != NULL && test->has_utility) {
        return sf_evaluate_utility_model(net, test);
    }

    nn_real *candidate_inputs =
        calloc(SF_ACTION_COUNT * SF_FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(SF_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(SF_ACTION_COUNT, sizeof(*probabilities));
    size_t *ranked = calloc(SF_ACTION_COUNT, sizeof(*ranked));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL || ranked == NULL) {
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        free(ranked);
        return metrics;
    }

    for (size_t i = 0; i < test->count; ++i) {
        const SFTraceRow *trace = &test->items[i];
        size_t candidate_count = sf_fill_candidates(trace, candidate_inputs);
        if (!nn_score_action_rows(net, candidate_inputs, candidate_count, scores, probabilities) ||
            !nn_rank_action_scores(scores, candidate_count, ranked)) {
            continue;
        }

        size_t target = (size_t)trace->action;
        size_t selected = ranked[0];
        size_t rank = candidate_count;
        for (size_t j = 0; j < candidate_count; ++j) {
            if (ranked[j] == target) {
                rank = j;
                break;
            }
        }
        if (rank >= candidate_count) {
            continue;
        }

        double confidence = (double)probabilities[selected];
        size_t bin = confidence >= 1.0 ? SF_CALIBRATION_BINS - 1u
                                       : (size_t)(confidence * (double)SF_CALIBRATION_BINS);
        if (bin >= SF_CALIBRATION_BINS) {
            bin = SF_CALIBRATION_BINS - 1u;
        }

        ++metrics.samples;
        metrics.top1 += rank == 0 ? 1u : 0u;
        metrics.top3 += rank < 3u ? 1u : 0u;
        metrics.selected_in_mask += (trace->candidate_mask & (UINT32_C(1) << selected)) != 0u ? 1u : 0u;
        metrics.teacher_in_mask += (trace->candidate_mask & (UINT32_C(1) << target)) != 0u ? 1u : 0u;
        ++metrics.target_counts[target];
        ++metrics.selected_counts[selected];
        ++metrics.confusion[target][selected];
        metrics.nll += -log(sf_safe_probability((double)probabilities[target]));
        metrics.mean_rank += (double)rank + 1.0;
        metrics.confidence += confidence;
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += rank == 0 ? 1.0 : 0.0;
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= sample_count;
    metrics.mean_rank /= sample_count;
    metrics.confidence /= sample_count;
    for (size_t bin = 0; bin < SF_CALIBRATION_BINS; ++bin) {
        if (metrics.bin_count[bin] == 0) {
            continue;
        }
        metrics.bin_confidence[bin] /= (double)metrics.bin_count[bin];
        metrics.bin_accuracy[bin] /= (double)metrics.bin_count[bin];
        metrics.ece += fabs(metrics.bin_confidence[bin] - metrics.bin_accuracy[bin]) *
                       ((double)metrics.bin_count[bin] / sample_count);
    }

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free(ranked);
    return metrics;
}

static int sf_save_checkpoint(NeuralNet *net,
                              const char *path,
                              int utility_mode,
                              int live_features,
                              char out_hash[65])
{
    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = live_features ? 2 : 1;
    strcpy(meta.parent_model_hash, "none");
    strcpy(meta.training_trace_root,
           utility_mode ? "signal_flight_counterfactual_csv"
                        : "signal_flight_trace_csv");
    strcpy(meta.notes,
           live_features
               ? "Signal flight WASD scorer trained from replay-certified utility with live-observable features"
               : (utility_mode
                      ? "Signal flight WASD scorer trained from replay-certified counterfactual utility"
                      : "Signal flight WASD scorer trained from flight_trace CSV"));
    strcpy(meta.feature_set, live_features ? "signal-flight-live-v2" : "signal-flight-wasd-v1");
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

static void sf_json_string(FILE *fp, const char *text)
{
    fputc('"', fp);
    if (text != NULL) {
        for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
            switch (*p) {
            case '"':
                fputs("\\\"", fp);
                break;
            case '\\':
                fputs("\\\\", fp);
                break;
            case '\b':
                fputs("\\b", fp);
                break;
            case '\f':
                fputs("\\f", fp);
                break;
            case '\n':
                fputs("\\n", fp);
                break;
            case '\r':
                fputs("\\r", fp);
                break;
            case '\t':
                fputs("\\t", fp);
                break;
            default:
                if (*p < 0x20u) {
                    fprintf(fp, "\\u%04x", (unsigned int)*p);
                } else {
                    fputc((int)*p, fp);
                }
                break;
            }
        }
    }
    fputc('"', fp);
}

static double sf_rate(size_t numerator, size_t denominator)
{
    return denominator == 0 ? 0.0 : (double)numerator / (double)denominator;
}

static void sf_print_action_counts(FILE *fp, const size_t counts[SF_ACTION_COUNT])
{
    fputc('{', fp);
    for (size_t i = 0; i < SF_ACTION_COUNT; ++i) {
        if (i > 0) {
            fputc(',', fp);
        }
        sf_json_string(fp, SF_ACTION_NAMES[i]);
        fprintf(fp, ":%zu", counts[i]);
    }
    fputc('}', fp);
}

static void sf_write_summary(FILE *fp,
                             const SFConfig *config,
                             const char *model_hash,
                             const SFDataset *train,
                             const SFDataset *test,
                             const NeuralNet *net,
                             const SFMetrics *metrics)
{
    fprintf(fp, "{\"schema\":");
    sf_json_string(fp, SF_SUMMARY_SCHEMA);
    fprintf(fp, ",\"task\":\"signal_flight\"");
    fprintf(fp, ",\"trace_schema\":");
    sf_json_string(fp, train->has_utility ? SF_COUNTERFACTUAL_TRACE_SCHEMA : SF_TRACE_SCHEMA);
    fprintf(fp, ",\"training_mode\":");
    sf_json_string(fp, train->has_utility ? "counterfactual_utility" : "teacher_imitation");
    fprintf(fp, ",\"feature_mode\":");
    sf_json_string(fp, config->live_features ? "live_observable" : "replay_outcome");
    fprintf(fp, ",\"model_hash\":");
    sf_json_string(fp, model_hash);
    fprintf(fp, ",\"train_trace\":");
    sf_json_string(fp, config->train_path);
    fprintf(fp, ",\"test_trace\":");
    sf_json_string(fp, config->test_path);
    fprintf(fp, ",\"train_rows\":%zu", train->count);
    fprintf(fp, ",\"test_rows\":%zu", test->count);
    fprintf(fp, ",\"epochs\":%zu", config->epochs);
    fprintf(fp, ",\"learning_rate\":%.9g", (double)config->learning_rate);
    fprintf(fp, ",\"seed\":%" PRIu64, config->seed);
    fprintf(fp, ",\"feature_count\":%u", (unsigned)SF_FEATURE_COUNT);
    fprintf(fp, ",\"candidate_count\":%u", (unsigned)SF_ACTION_COUNT);
    fprintf(fp, ",\"parameter_count\":%zu", nn_parameter_count(net));
    fprintf(fp, ",\"forward_flops\":%zu", nn_forward_flop_count(net));
    fprintf(fp, ",\"top1\":%.9f", sf_rate(metrics->top1, metrics->samples));
    fprintf(fp, ",\"top3\":%.9f", sf_rate(metrics->top3, metrics->samples));
    fprintf(fp, ",\"nll\":%.9f", metrics->nll);
    fprintf(fp, ",\"ece\":%.9f", metrics->ece);
    fprintf(fp, ",\"mean_rank\":%.9f", metrics->mean_rank);
    fprintf(fp, ",\"mean_confidence\":%.9f", metrics->confidence);
    fprintf(fp, ",\"selected_candidate_mask_rate\":%.9f",
            sf_rate(metrics->selected_in_mask, metrics->samples));
    fprintf(fp, ",\"target_candidate_mask_rate\":%.9f",
            sf_rate(metrics->teacher_in_mask, metrics->samples));
    if (metrics->utility_groups > 0u) {
        fprintf(fp, ",\"utility_groups\":%zu", metrics->utility_groups);
        fprintf(fp, ",\"selected_utility\":%.9f", metrics->selected_utility);
        fprintf(fp, ",\"oracle_utility\":%.9f", metrics->oracle_utility);
        fprintf(fp, ",\"regret\":%.9f", metrics->regret);
        fprintf(fp, ",\"selected_damage_rate\":%.9f",
                sf_rate(metrics->selected_damage, metrics->samples));
        fprintf(fp, ",\"oracle_damage_rate\":%.9f",
                sf_rate(metrics->oracle_damage, metrics->samples));
        fprintf(fp, ",\"selected_death_rate\":%.9f",
                sf_rate(metrics->selected_death, metrics->samples));
        fprintf(fp, ",\"oracle_death_rate\":%.9f",
                sf_rate(metrics->oracle_death, metrics->samples));
        fprintf(fp, ",\"baseline_name\":");
        sf_json_string(fp, "goal_heading_no_safety");
        fprintf(fp, ",\"baseline_top1\":%.9f",
                sf_rate(metrics->baseline_top1, metrics->baseline_samples));
        fprintf(fp, ",\"baseline_selected_utility\":%.9f",
                metrics->baseline_selected_utility);
        fprintf(fp, ",\"baseline_oracle_utility\":%.9f",
                metrics->baseline_oracle_utility);
        fprintf(fp, ",\"baseline_regret\":%.9f", metrics->baseline_regret);
        fprintf(fp, ",\"baseline_damage_rate\":%.9f",
                sf_rate(metrics->baseline_damage, metrics->baseline_samples));
        fprintf(fp, ",\"baseline_death_rate\":%.9f",
                sf_rate(metrics->baseline_death, metrics->baseline_samples));
        fprintf(fp, ",\"neural_utility_delta_vs_baseline\":%.9f",
                metrics->selected_utility - metrics->baseline_selected_utility);
        fprintf(fp, ",\"neural_regret_delta_vs_baseline\":%.9f",
                metrics->regret - metrics->baseline_regret);
    }
    fprintf(fp, ",\"target_action_counts\":");
    sf_print_action_counts(fp, metrics->target_counts);
    fprintf(fp, ",\"selected_action_counts\":");
    sf_print_action_counts(fp, metrics->selected_counts);
    fprintf(fp, ",\"confusion\":[");
    for (size_t target = 0; target < SF_ACTION_COUNT; ++target) {
        if (target > 0) {
            fputc(',', fp);
        }
        fputc('[', fp);
        for (size_t selected = 0; selected < SF_ACTION_COUNT; ++selected) {
            if (selected > 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%zu", metrics->confusion[target][selected]);
        }
        fputc(']', fp);
    }
    fprintf(fp, "]}\n");
}

static int sf_write_summary_file(const char *path,
                                 const SFConfig *config,
                                 const char *model_hash,
                                 const SFDataset *train,
                                 const SFDataset *test,
                                 const NeuralNet *net,
                                 const SFMetrics *metrics)
{
    if (path == NULL) {
        return 1;
    }
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "failed to open Signal flight summary for write: %s\n", path);
        return 0;
    }
    sf_write_summary(fp, config, model_hash, train, test, net, metrics);
    int ok = ferror(fp) == 0;
    fclose(fp);
    return ok;
}

static void sf_print_metrics_line(const SFMetrics *metrics)
{
    if (metrics->utility_groups > 0u) {
        printf("signal_flight counterfactual top1=%.2f%% top3=%.2f%% regret=%.4f baseline_regret=%.4f selected_utility=%.4f baseline_utility=%.4f oracle_utility=%.4f selected_death=%.2f%% samples=%zu\n",
               100.0 * sf_rate(metrics->top1, metrics->samples),
               100.0 * sf_rate(metrics->top3, metrics->samples),
               metrics->regret,
               metrics->baseline_regret,
               metrics->selected_utility,
               metrics->baseline_selected_utility,
               metrics->oracle_utility,
               100.0 * sf_rate(metrics->selected_death, metrics->samples),
               metrics->samples);
    } else {
        printf("signal_flight top1=%.2f%% top3=%.2f%% nll=%.4f ece=%.4f candidate_selected=%.2f%% mean_rank=%.2f samples=%zu\n",
               100.0 * sf_rate(metrics->top1, metrics->samples),
               100.0 * sf_rate(metrics->top3, metrics->samples),
               metrics->nll,
               metrics->ece,
               100.0 * sf_rate(metrics->selected_in_mask, metrics->samples),
               metrics->mean_rank,
               metrics->samples);
    }
}

static void sf_print_usage(FILE *stream)
{
    fputs("usage: signal_flight [--schema] --train path --test path "
          "[--epochs count] [--max-train count] [--max-test count] "
          "[--learning-rate rate] [--seed seed] [--save path] [--summary path] "
          "[--live-features]\n",
          stream);
}

static int sf_parse_config(int argc, char **argv, SFConfig *config, int *schema_only)
{
    if (config == NULL || schema_only == NULL) {
        return 0;
    }
    memset(config, 0, sizeof(*config));
    config->epochs = 3u;
    config->max_train = 0u;
    config->max_test = 0u;
    config->seed = 4242u;
    config->learning_rate = (nn_real)0.004;
    *schema_only = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--schema") == 0) {
            *schema_only = 1;
        } else if (strcmp(argv[i], "--train") == 0 && i + 1 < argc) {
            config->train_path = argv[++i];
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            config->test_path = argv[++i];
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!sf_parse_size(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-train") == 0 && i + 1 < argc) {
            if (!sf_parse_size(argv[++i], &config->max_train)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-test") == 0 && i + 1 < argc) {
            if (!sf_parse_size(argv[++i], &config->max_test)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            if (!sf_parse_real(argv[++i], &config->learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!sf_parse_u64(argv[++i], &config->seed)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--summary") == 0 && i + 1 < argc) {
            config->summary_path = argv[++i];
        } else if (strcmp(argv[i], "--live-features") == 0) {
            config->live_features = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            sf_print_usage(stdout);
            exit(EXIT_SUCCESS);
        } else {
            return 0;
        }
    }

    if (*schema_only) {
        return 1;
    }
    return config->train_path != NULL && config->test_path != NULL && config->epochs > 0;
}

int main(int argc, char **argv)
{
    SFConfig config;
    int schema_only = 0;
    if (!sf_parse_config(argc, argv, &config, &schema_only)) {
        sf_print_usage(stderr);
        return EXIT_FAILURE;
    }
    if (schema_only) {
        return nn_experiment_write_schema_json(stdout, &SF_EXPERIMENT_SPEC)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }

    SFDataset train = {0};
    SFDataset test = {0};
    if (!sf_load_trace_csv(config.train_path, config.max_train, &train) ||
        !sf_load_trace_csv(config.test_path, config.max_test, &test)) {
        sf_dataset_free(&train);
        sf_dataset_free(&test);
        return EXIT_FAILURE;
    }
    if (train.has_utility != test.has_utility) {
        fprintf(stderr, "train/test Signal traces must use the same supervision mode\n");
        sf_dataset_free(&train);
        sf_dataset_free(&test);
        return EXIT_FAILURE;
    }

    NeuralNet *net = sf_create_model((unsigned int)(config.seed & UINT64_C(0xffffffff)));
    if (net == NULL) {
        sf_dataset_free(&train);
        sf_dataset_free(&test);
        return EXIT_FAILURE;
    }

    g_sf_live_features = config.live_features;

    if (!sf_train_model(net, &train, &config)) {
        nn_free(net);
        sf_dataset_free(&train);
        sf_dataset_free(&test);
        return EXIT_FAILURE;
    }

    char model_hash[65] = "unsaved";
    if (config.save_path != NULL) {
        if (!sf_save_checkpoint(net,
                                config.save_path,
                                train.has_utility,
                                config.live_features,
                                model_hash)) {
            fprintf(stderr, "failed to save Signal flight checkpoint: %s\n", config.save_path);
            nn_free(net);
            sf_dataset_free(&train);
            sf_dataset_free(&test);
            return EXIT_FAILURE;
        }
        printf("saved signal_flight checkpoint: %s hash=%s\n", config.save_path, model_hash);
    }

    SFMetrics metrics = sf_evaluate_model(net, &test);
    sf_write_summary(stdout, &config, model_hash, &train, &test, net, &metrics);
    if (!sf_write_summary_file(config.summary_path,
                               &config,
                               model_hash,
                               &train,
                               &test,
                               net,
                               &metrics)) {
        nn_free(net);
        sf_dataset_free(&train);
        sf_dataset_free(&test);
        return EXIT_FAILURE;
    }
    sf_print_metrics_line(&metrics);

    nn_free(net);
    sf_dataset_free(&train);
    sf_dataset_free(&test);
    return metrics.samples == 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
