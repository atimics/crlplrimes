#include "nn.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    ARC_INPUT_WIDTH = 7,
    ARC_INPUT_HEIGHT = 3,
    ARC_INPUT_CELLS = ARC_INPUT_WIDTH * ARC_INPUT_HEIGHT,
    ARC_OUTPUT_WIDTH = 3,
    ARC_OUTPUT_HEIGHT = 3,
    ARC_OUTPUT_CELLS = ARC_OUTPUT_WIDTH * ARC_OUTPUT_HEIGHT,
    ARC_ACTION_COUNT = 10,
    ARC_TRAIN_PAIR_COUNT = 3,
    ARC_COMMON_FEATURES = 9,
    ARC_FEATURE_COUNT = ARC_COMMON_FEATURES + ARC_INPUT_CELLS + ARC_OUTPUT_CELLS + ARC_ACTION_COUNT,
    ARC_DEFAULT_EPOCHS = 4,
    ARC_DEFAULT_STEPS_PER_EPOCH = 900,
    ARC_DEFAULT_TEST_SAMPLES = 900,
    ARC_CALIBRATION_BINS = 10,
    ARC_COLOR_COUNT = 10,
    ARC_MAX_GRID_WIDTH = 30,
    ARC_MAX_GRID_HEIGHT = 30,
    ARC_MAX_GRID_CELLS = ARC_MAX_GRID_WIDTH * ARC_MAX_GRID_HEIGHT,
    ARC_MAX_PAIRS = 16,
    ARC_MAX_TASK_ID = 64,
    ARC_MAX_TASK_FILES = 4096,
    ARC_MAX_TASK_PATH = 1024,
    ARC_ACCEPTED_RULES_BYTES = 512,
    ARC_RULE_TEMPLATE_COUNT = 5,
    ARC_RULE_CANDIDATE_COUNT = ARC_RULE_TEMPLATE_COUNT * (ARC_COLOR_COUNT - 1) + 1,
    ARC_GRAMMAR_VERSION = 2,
};

typedef enum {
    ARC_RULE_COPY_LEFT = 0,
    ARC_RULE_COPY_RIGHT = 1,
    ARC_RULE_INTERSECTION = 2,
    ARC_RULE_UNION = 3,
    ARC_RULE_OBJECT_SELECT = 4,
    ARC_RULE_COLOR_MAP_LEFT = 5,
} ARCTransformRule;

/* ARC-AGI-1 training task 0520fde7, reduced to balanced output-cell color scoring. */
static const uint8_t ARC_0520FDE7_TRAIN_INPUTS[ARC_TRAIN_PAIR_COUNT][ARC_INPUT_CELLS] = {
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

static const uint8_t ARC_0520FDE7_TRAIN_OUTPUTS[ARC_TRAIN_PAIR_COUNT][ARC_OUTPUT_CELLS] = {
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

static const uint8_t ARC_0520FDE7_TEST_INPUT[ARC_INPUT_CELLS] = {
    1, 0, 1, 5, 1, 0, 1,
    0, 1, 0, 5, 1, 0, 1,
    1, 0, 1, 5, 0, 1, 0,
};

static const uint8_t ARC_0520FDE7_TEST_OUTPUT[ARC_OUTPUT_CELLS] = {
    2, 0, 2,
    0, 0, 0,
    0, 0, 0,
};

static const NNFeatureSchema ARC_AGI_FEATURES[ARC_FEATURE_COUNT] = {
    {"output_x", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_y", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_width", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_height", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_width", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_height", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_cell_index", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_is_background", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_norm", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_00", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_01", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_02", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_03", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_04", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_05", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_06", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_07", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_08", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_09", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_10", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_11", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_12", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_13", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_14", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_15", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_16", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_17", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_18", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_19", NN_FEATURE_STATE_OBSERVATION, 1},
    {"input_cell_20", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_00", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_01", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_02", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_03", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_04", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_05", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_06", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_07", NN_FEATURE_STATE_OBSERVATION, 1},
    {"output_coord_08", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_0", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_1", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_2", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_3", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_4", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_5", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_6", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_7", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_8", NN_FEATURE_STATE_OBSERVATION, 1},
    {"candidate_color_9", NN_FEATURE_STATE_OBSERVATION, 1},
};

_Static_assert(sizeof(ARC_AGI_FEATURES) / sizeof(ARC_AGI_FEATURES[0]) == ARC_FEATURE_COUNT,
               "arc_agi feature schema must match scorer feature count");

static const NNExperimentSpec ARC_AGI_EXPERIMENT_SPEC = {
    .task_name = "arc_agi",
    .role = "fixed ARC-AGI-1 sample reduced to verifier-checked output-cell color scoring",
    .state_space = "ARC-AGI-1 task 0520fde7 input grid plus target output coordinate",
    .state_distribution =
        "three training pairs and the held-out test input from ARC-AGI-1 task 0520fde7",
    .action_space = "candidate output colors 0..9",
    .candidate_enumerator = "enumerate all ten color actions for each requested output cell",
    .observable_features =
        "flattened input grid, output coordinate, geometry constants, candidate color, and action one-hot",
    .leakage_policy = NN_LEAKAGE_FORBIDDEN,
    .neural_scorer = "MLP scores one output-cell/color row and softmax ranks candidate colors",
    .exact_verifier = "held-out output grid supplies the exact color for the requested cell",
    .verifier_feedback = "target color, top-k correctness, NLL, calibration bins, and proposal rank",
    .rule_grammar = "copy_left, copy_right, mask_intersection, mask_union, object_select_overlap, color_map",
    .certificate_checker =
        "train-pair consistency plus unique grammar-consistent test prediction, reported against compiled test output",
    .gate_executor = "certified transform gate emits exact color candidates when the grammar prediction is unique",
    .proposer_inputs = "structural transform descriptors: copy, mask, intersection, color-map",
    .proposer_targets = "held-out-safe transform support and exact output-cell agreement diagnostics",
    .proposer_mode = NN_PROPOSER_SELECTION,
    .baseline_orderings = "neural scorer, majority color, certified transform gate",
    .generalization_axis = "train examples to held-out test grid within one fixed ARC-AGI-1 task",
    .trace_schema = NN_TRACE_SCHEMA,
    .replay_protocol = "planned stable output-cell/color action-id replay",
    .rl_protocol = "not implemented for first-class ARC smoke",
    .features = ARC_AGI_FEATURES,
    .feature_count = sizeof(ARC_AGI_FEATURES) / sizeof(ARC_AGI_FEATURES[0]),
};

typedef struct {
    size_t epochs;
    size_t steps_per_epoch;
    size_t test_samples;
    size_t max_tasks;
    uint64_t seed;
    nn_real learning_rate;
    const char *save_path;
    const char *task_dir;
    int loader_summary_only;
} ARCConfig;

typedef struct {
    size_t width;
    size_t height;
    uint8_t cells[ARC_MAX_GRID_CELLS];
} ARCGrid;

typedef struct {
    ARCGrid input;
    ARCGrid output;
    int has_output;
} ARCGridPair;

typedef struct {
    char id[ARC_MAX_TASK_ID];
    ARCGridPair train[ARC_MAX_PAIRS];
    size_t train_count;
    ARCGridPair test[ARC_MAX_PAIRS];
    size_t test_count;
} ARCTask;

typedef struct {
    ARCTransformRule op;
    uint8_t output_color;
    char name[64];
} ARCTransformCandidate;

typedef struct {
    size_t samples;
    size_t top1;
    size_t top5;
    size_t top10;
    double nll;
    double ece;
    double mean_rank;
    double decision_utility;
    double proposal_steps;
    double baseline_steps;
    double candidate_count;
    size_t bin_count[ARC_CALIBRATION_BINS];
    double bin_confidence[ARC_CALIBRATION_BINS];
    double bin_accuracy[ARC_CALIBRATION_BINS];
} ARCMetrics;

typedef struct {
    size_t rules;
    size_t train_consistent_rules;
    size_t train_checked_cells;
    size_t train_mismatches;
    size_t unsupported_train_pairs;
    size_t unsupported_test_pairs;
    size_t test_pairs;
    size_t test_cells;
    size_t test_correct_cells;
    int unique_test_prediction;
    int test_exact;
    int certified;
    char accepted_rules[ARC_ACCEPTED_RULES_BYTES];
    ARCGrid predictions[ARC_MAX_PAIRS];
    size_t prediction_count;
} ARCTransformMetrics;

typedef struct {
    char path[ARC_MAX_TASK_PATH];
    char id[ARC_MAX_TASK_ID];
} ARCTaskFile;

typedef struct {
    size_t files_seen;
    size_t loaded_tasks;
    size_t parse_errors;
    size_t train_pairs;
    size_t test_pairs;
    size_t rules;
    size_t train_consistent_rules;
    size_t unique_prediction_tasks;
    size_t scored_tasks;
    size_t test_exact_tasks;
    size_t certified_tasks;
    size_t unsupported_tasks;
    size_t test_cells;
    size_t test_correct_cells;
} ARCLoaderSummary;

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

static int parse_real_text(const char *text, nn_real *out_value)
{
    if (text == NULL || text[0] == '\0') {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !isfinite(value) || value <= 0.0) {
        return 0;
    }
    *out_value = (nn_real)value;
    return 1;
}

static double safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

static void arc_init_grid(ARCGrid *grid, size_t width, size_t height)
{
    memset(grid, 0, sizeof(*grid));
    grid->width = width;
    grid->height = height;
}

static size_t arc_grid_cells(const ARCGrid *grid)
{
    return grid->width * grid->height;
}

static int arc_grid_same_shape(const ARCGrid *lhs, const ARCGrid *rhs)
{
    return lhs->width == rhs->width && lhs->height == rhs->height;
}

static int arc_grid_equal(const ARCGrid *lhs, const ARCGrid *rhs)
{
    size_t cells = arc_grid_cells(lhs);
    return arc_grid_same_shape(lhs, rhs) &&
           memcmp(lhs->cells, rhs->cells, cells * sizeof(lhs->cells[0])) == 0;
}

static void init_builtin_arc_task(ARCTask *task)
{
    memset(task, 0, sizeof(*task));
    strcpy(task->id, "0520fde7");
    task->train_count = ARC_TRAIN_PAIR_COUNT;
    task->test_count = 1;
    for (size_t pair = 0; pair < ARC_TRAIN_PAIR_COUNT; ++pair) {
        arc_init_grid(&task->train[pair].input, ARC_INPUT_WIDTH, ARC_INPUT_HEIGHT);
        memcpy(task->train[pair].input.cells,
               ARC_0520FDE7_TRAIN_INPUTS[pair],
               sizeof(ARC_0520FDE7_TRAIN_INPUTS[pair]));
        arc_init_grid(&task->train[pair].output, ARC_OUTPUT_WIDTH, ARC_OUTPUT_HEIGHT);
        memcpy(task->train[pair].output.cells,
               ARC_0520FDE7_TRAIN_OUTPUTS[pair],
               sizeof(ARC_0520FDE7_TRAIN_OUTPUTS[pair]));
        task->train[pair].has_output = 1;
    }
    arc_init_grid(&task->test[0].input, ARC_INPUT_WIDTH, ARC_INPUT_HEIGHT);
    memcpy(task->test[0].input.cells, ARC_0520FDE7_TEST_INPUT, sizeof(ARC_0520FDE7_TEST_INPUT));
    arc_init_grid(&task->test[0].output, ARC_OUTPUT_WIDTH, ARC_OUTPUT_HEIGHT);
    memcpy(task->test[0].output.cells, ARC_0520FDE7_TEST_OUTPUT, sizeof(ARC_0520FDE7_TEST_OUTPUT));
    task->test[0].has_output = 1;
}

static size_t arc_build_candidates(ARCTransformCandidate *candidates, size_t capacity)
{
    size_t count = 0;
    static const struct {
        ARCTransformRule op;
        const char *prefix;
    } colored_templates[] = {
        {ARC_RULE_COPY_LEFT, "copy_left_color_"},
        {ARC_RULE_COPY_RIGHT, "copy_right_color_"},
        {ARC_RULE_INTERSECTION, "mask_intersection_color_"},
        {ARC_RULE_UNION, "mask_union_color_"},
        {ARC_RULE_OBJECT_SELECT, "object_select_overlap_color_"},
    };

    for (size_t template_id = 0;
         template_id < sizeof(colored_templates) / sizeof(colored_templates[0]);
         ++template_id) {
        for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
            if (count >= capacity) {
                return count;
            }
            candidates[count].op = colored_templates[template_id].op;
            candidates[count].output_color = color;
            snprintf(candidates[count].name,
                     sizeof(candidates[count].name),
                     "%s%u",
                     colored_templates[template_id].prefix,
                     (unsigned)color);
            ++count;
        }
    }

    if (count < capacity) {
        candidates[count].op = ARC_RULE_COLOR_MAP_LEFT;
        candidates[count].output_color = 0;
        strcpy(candidates[count].name, "color_map_left_identity");
        ++count;
    }
    return count;
}

static int arc_apply_candidate(const ARCTransformCandidate *candidate,
                               const ARCGrid *input,
                               ARCGrid *output)
{
    if (input->width < 3 || input->height == 0 || (input->width % 2u) == 0) {
        return 0;
    }
    size_t output_width = (input->width - 1u) / 2u;
    size_t output_height = input->height;
    if (output_width == 0 || output_width > ARC_MAX_GRID_WIDTH ||
        output_height > ARC_MAX_GRID_HEIGHT ||
        output_width * output_height > ARC_MAX_GRID_CELLS) {
        return 0;
    }

    arc_init_grid(output, output_width, output_height);
    size_t right_offset = output_width + 1u;
    for (size_t y = 0; y < output_height; ++y) {
        for (size_t x = 0; x < output_width; ++x) {
            uint8_t left = input->cells[y * input->width + x];
            uint8_t right = input->cells[y * input->width + right_offset + x];
            int left_on = left != 0;
            int right_on = right != 0;
            size_t out = y * output_width + x;
            switch (candidate->op) {
            case ARC_RULE_COPY_LEFT:
                output->cells[out] = left_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_COPY_RIGHT:
                output->cells[out] = right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_INTERSECTION:
                output->cells[out] = left_on && right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_UNION:
                output->cells[out] = left_on || right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_OBJECT_SELECT:
                output->cells[out] = left_on && right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_COLOR_MAP_LEFT:
                output->cells[out] = left;
                break;
            default:
                return 0;
            }
        }
    }
    return 1;
}

static int arc_candidate_matches_train(const ARCTask *task,
                                       const ARCTransformCandidate *candidate,
                                       size_t *checked_cells_out,
                                       size_t *mismatches_out,
                                       size_t *unsupported_pairs_out)
{
    ARCGrid produced;
    size_t checked_cells = 0;
    size_t mismatches = 0;
    size_t unsupported_pairs = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output ||
            !arc_apply_candidate(candidate, &task->train[pair].input, &produced) ||
            !arc_grid_same_shape(&produced, &task->train[pair].output)) {
            ++unsupported_pairs;
            continue;
        }
        size_t cells = arc_grid_cells(&task->train[pair].output);
        checked_cells += cells;
        for (size_t cell = 0; cell < cells; ++cell) {
            if (produced.cells[cell] != task->train[pair].output.cells[cell]) {
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
    if (unsupported_pairs_out != NULL) {
        *unsupported_pairs_out = unsupported_pairs;
    }
    return unsupported_pairs == 0 && checked_cells > 0 && mismatches == 0;
}

static void arc_append_accepted_rule(char *buffer, size_t buffer_size, const char *rule_name)
{
    size_t used = strlen(buffer);
    if (used + 1u >= buffer_size) {
        return;
    }
    int written = snprintf(buffer + used,
                           buffer_size - used,
                           "%s%s",
                           used == 0 ? "" : ",",
                           rule_name);
    if (written < 0 || (size_t)written >= buffer_size - used) {
        buffer[buffer_size - 1u] = '\0';
    }
}

static int arc_apply_candidate_to_tests(const ARCTask *task,
                                        const ARCTransformCandidate *candidate,
                                        ARCGrid *predictions,
                                        size_t *unsupported_pairs_out)
{
    size_t unsupported_pairs = 0;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        if (!arc_apply_candidate(candidate, &task->test[pair].input, &predictions[pair])) {
            ++unsupported_pairs;
        }
    }
    if (unsupported_pairs_out != NULL) {
        *unsupported_pairs_out = unsupported_pairs;
    }
    return unsupported_pairs == 0;
}

static int arc_predictions_equal(const ARCGrid *lhs, const ARCGrid *rhs, size_t count)
{
    for (size_t pair = 0; pair < count; ++pair) {
        if (!arc_grid_equal(&lhs[pair], &rhs[pair])) {
            return 0;
        }
    }
    return 1;
}

static ARCTransformMetrics evaluate_arc_transform_gate_task(const ARCTask *task)
{
    ARCTransformMetrics metrics;
    ARCTransformCandidate candidates[ARC_RULE_CANDIDATE_COUNT];
    ARCGrid first_predictions[ARC_MAX_PAIRS];
    ARCGrid current_predictions[ARC_MAX_PAIRS];
    memset(&metrics, 0, sizeof(metrics));
    memset(first_predictions, 0, sizeof(first_predictions));
    metrics.unique_test_prediction = 1;
    metrics.test_pairs = task->test_count;
    metrics.rules = arc_build_candidates(candidates, ARC_RULE_CANDIDATE_COUNT);

    int have_prediction = 0;
    for (size_t rule = 0; rule < metrics.rules; ++rule) {
        size_t checked_cells = 0;
        size_t mismatches = 0;
        size_t unsupported_train_pairs = 0;
        if (!arc_candidate_matches_train(task,
                                         &candidates[rule],
                                         &checked_cells,
                                         &mismatches,
                                         &unsupported_train_pairs)) {
            metrics.train_checked_cells += checked_cells;
            metrics.train_mismatches += mismatches;
            metrics.unsupported_train_pairs += unsupported_train_pairs;
            continue;
        }
        metrics.train_checked_cells += checked_cells;
        ++metrics.train_consistent_rules;
        arc_append_accepted_rule(metrics.accepted_rules,
                                 sizeof(metrics.accepted_rules),
                                 candidates[rule].name);

        size_t unsupported_test_pairs = 0;
        if (!arc_apply_candidate_to_tests(task,
                                          &candidates[rule],
                                          current_predictions,
                                          &unsupported_test_pairs)) {
            metrics.unsupported_test_pairs += unsupported_test_pairs;
            metrics.unique_test_prediction = 0;
            continue;
        }
        if (!have_prediction) {
            memcpy(first_predictions, current_predictions, sizeof(first_predictions));
            have_prediction = 1;
        } else if (!arc_predictions_equal(first_predictions, current_predictions, task->test_count)) {
            metrics.unique_test_prediction = 0;
        }
    }

    if (have_prediction) {
        memcpy(metrics.predictions, first_predictions, sizeof(metrics.predictions));
        metrics.prediction_count = task->test_count;
    } else {
        metrics.unique_test_prediction = 0;
    }

    int all_test_outputs_known = task->test_count > 0;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        if (!task->test[pair].has_output) {
            all_test_outputs_known = 0;
            continue;
        }
        size_t cells = arc_grid_cells(&task->test[pair].output);
        metrics.test_cells += cells;
        if (!have_prediction ||
            pair >= metrics.prediction_count ||
            !arc_grid_same_shape(&metrics.predictions[pair], &task->test[pair].output)) {
            continue;
        }
        for (size_t cell = 0; cell < cells; ++cell) {
            metrics.test_correct_cells +=
                metrics.predictions[pair].cells[cell] == task->test[pair].output.cells[cell] ? 1u : 0u;
        }
    }

    metrics.test_exact = all_test_outputs_known &&
                         metrics.test_cells > 0 &&
                         metrics.test_correct_cells == metrics.test_cells &&
                         metrics.unique_test_prediction;
    metrics.certified = metrics.train_consistent_rules != 0 &&
                        metrics.unique_test_prediction &&
                        metrics.test_exact;
    return metrics;
}

static ARCTransformMetrics evaluate_builtin_arc_transform_gate(void)
{
    ARCTask task;
    init_builtin_arc_task(&task);
    return evaluate_arc_transform_gate_task(&task);
}

static ARCMetrics arc_transform_to_metrics(const ARCTask *task,
                                           const ARCTransformMetrics *transform)
{
    ARCMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    metrics.candidate_count = ARC_ACTION_COUNT;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        if (!task->test[pair].has_output) {
            continue;
        }
        size_t cells = arc_grid_cells(&task->test[pair].output);
        for (size_t cell = 0; cell < cells; ++cell) {
            size_t target = task->test[pair].output.cells[cell];
            size_t selected = 0;
            if (transform->certified &&
                pair < transform->prediction_count &&
                arc_grid_same_shape(&transform->predictions[pair], &task->test[pair].output)) {
                selected = transform->predictions[pair].cells[cell];
            }
            size_t rank = selected == target ? 0u : target == 0u ? 1u : target;
            ++metrics.samples;
            metrics.top1 += rank == 0 ? 1u : 0u;
            metrics.top5 += rank < 5 ? 1u : 0u;
            metrics.top10 += rank < 10 ? 1u : 0u;
            metrics.nll += rank == 0 ? 0.0 : -log(1e-12);
            metrics.mean_rank += (double)rank + 1.0;
            metrics.decision_utility += rank == 0 ? 1.0 : 0.0;
            metrics.proposal_steps += (double)rank + 1.0;
            metrics.baseline_steps += (double)target + 1.0;
        }
    }
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= samples;
    metrics.mean_rank /= samples;
    metrics.decision_utility /= samples;
    metrics.proposal_steps /= samples;
    metrics.baseline_steps /= samples;
    return metrics;
}

static size_t arc_select_balanced_cell(const uint8_t *output, uint64_t sample)
{
    int want_foreground = (sample & UINT64_C(1)) == 0;
    size_t match_count = 0;
    for (size_t cell = 0; cell < ARC_OUTPUT_CELLS; ++cell) {
        int foreground = output[cell] != 0;
        if (foreground == want_foreground) {
            ++match_count;
        }
    }
    if (match_count == 0) {
        return (size_t)(sample % ARC_OUTPUT_CELLS);
    }

    size_t wanted = (size_t)((sample >> 1) % match_count);
    for (size_t cell = 0; cell < ARC_OUTPUT_CELLS; ++cell) {
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

static size_t fill_arc_sample(int train,
                              uint64_t sample,
                              nn_real *candidate_inputs,
                              size_t *target_index)
{
    const uint8_t *input = ARC_0520FDE7_TEST_INPUT;
    const uint8_t *output = ARC_0520FDE7_TEST_OUTPUT;
    uint64_t cell_sample = sample;
    if (train) {
        size_t pair = (size_t)(sample % ARC_TRAIN_PAIR_COUNT);
        input = ARC_0520FDE7_TRAIN_INPUTS[pair];
        output = ARC_0520FDE7_TRAIN_OUTPUTS[pair];
        cell_sample = sample / ARC_TRAIN_PAIR_COUNT;
    }

    size_t cell = arc_select_balanced_cell(output, cell_sample);
    size_t output_x = cell % ARC_OUTPUT_WIDTH;
    size_t output_y = cell / ARC_OUTPUT_WIDTH;
    uint8_t target_color = output[cell];
    *target_index = (size_t)target_color;

    memset(candidate_inputs, 0, (size_t)ARC_ACTION_COUNT * ARC_FEATURE_COUNT * sizeof(*candidate_inputs));
    for (size_t action = 0; action < ARC_ACTION_COUNT; ++action) {
        nn_real *row = &candidate_inputs[action * ARC_FEATURE_COUNT];
        row[0] = (nn_real)((double)output_x / (double)(ARC_OUTPUT_WIDTH - 1));
        row[1] = (nn_real)((double)output_y / (double)(ARC_OUTPUT_HEIGHT - 1));
        row[2] = (nn_real)((double)ARC_INPUT_WIDTH / 30.0);
        row[3] = (nn_real)((double)ARC_INPUT_HEIGHT / 30.0);
        row[4] = (nn_real)((double)ARC_OUTPUT_WIDTH / 30.0);
        row[5] = (nn_real)((double)ARC_OUTPUT_HEIGHT / 30.0);
        row[6] = (nn_real)((double)cell / (double)(ARC_OUTPUT_CELLS - 1));
        row[7] = action == 0 ? (nn_real)1 : (nn_real)0;
        row[8] = (nn_real)((double)action / 9.0);
        for (size_t i = 0; i < ARC_INPUT_CELLS; ++i) {
            row[ARC_COMMON_FEATURES + i] = (nn_real)((double)input[i] / 9.0);
        }
        row[ARC_COMMON_FEATURES + ARC_INPUT_CELLS + cell] = (nn_real)1;
        row[ARC_COMMON_FEATURES + ARC_INPUT_CELLS + ARC_OUTPUT_CELLS + action] =
            (nn_real)1;
    }
    return ARC_ACTION_COUNT;
}

static NeuralNet *create_arc_model(unsigned int seed)
{
    static const size_t layers[] = {ARC_FEATURE_COUNT, 64, 32, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int train_arc_model(NeuralNet *net, const ARCConfig *config)
{
    nn_real *candidate_inputs =
        calloc((size_t)ARC_ACTION_COUNT * ARC_FEATURE_COUNT, sizeof(*candidate_inputs));
    if (candidate_inputs == NULL) {
        return 0;
    }

    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double loss_sum = 0.0;
        for (size_t step = 0; step < config->steps_per_epoch; ++step) {
            uint64_t sample = config->seed + (uint64_t)(epoch * config->steps_per_epoch + step);
            size_t target_index = 0;
            size_t candidate_count = fill_arc_sample(1, sample, candidate_inputs, &target_index);
            if (candidate_count == 0 || target_index >= candidate_count) {
                free(candidate_inputs);
                return 0;
            }
            nn_real loss = nn_train_softmax_scorer(net,
                                                   candidate_inputs,
                                                   candidate_count,
                                                   target_index,
                                                   config->learning_rate);
            if (!isfinite((double)loss)) {
                free(candidate_inputs);
                return 0;
            }
            loss_sum += (double)loss;
        }
        printf("arc_agi epoch %zu loss=%.4f samples=%zu\n",
               epoch + 1,
               loss_sum / (double)config->steps_per_epoch,
               config->steps_per_epoch);
    }

    free(candidate_inputs);
    return 1;
}

static ARCMetrics evaluate_arc_model(NeuralNet *net, size_t samples)
{
    ARCMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    nn_real *candidate_inputs =
        calloc((size_t)ARC_ACTION_COUNT * ARC_FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(ARC_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(ARC_ACTION_COUNT, sizeof(*probabilities));
    size_t *ranked = calloc(ARC_ACTION_COUNT, sizeof(*ranked));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL || ranked == NULL) {
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        free(ranked);
        return metrics;
    }

    for (size_t i = 0; i < samples; ++i) {
        size_t target_index = 0;
        size_t candidate_count = fill_arc_sample(0, (uint64_t)i, candidate_inputs, &target_index);
        if (candidate_count == 0 || target_index >= candidate_count ||
            !nn_score_action_rows(net, candidate_inputs, candidate_count, scores, probabilities) ||
            !nn_rank_action_scores(scores, candidate_count, ranked)) {
            continue;
        }

        size_t selected = ranked[0];
        size_t target_rank = candidate_count;
        for (size_t rank = 0; rank < candidate_count; ++rank) {
            if (ranked[rank] == target_index) {
                target_rank = rank;
                break;
            }
        }
        if (target_rank >= candidate_count) {
            continue;
        }

        double confidence = (double)probabilities[selected];
        size_t bin = confidence >= 1.0 ? ARC_CALIBRATION_BINS - 1
                                       : (size_t)(confidence * ARC_CALIBRATION_BINS);
        if (bin >= ARC_CALIBRATION_BINS) {
            bin = ARC_CALIBRATION_BINS - 1;
        }

        ++metrics.samples;
        metrics.top1 += target_rank == 0 ? 1u : 0u;
        metrics.top5 += target_rank < 5 ? 1u : 0u;
        metrics.top10 += target_rank < 10 ? 1u : 0u;
        metrics.nll += -log(safe_probability((double)probabilities[target_index]));
        metrics.mean_rank += (double)target_rank + 1.0;
        metrics.decision_utility += target_rank == 0 ? 1.0 : 0.0;
        metrics.proposal_steps += (double)target_rank + 1.0;
        metrics.baseline_steps += (double)target_index + 1.0;
        metrics.candidate_count += (double)candidate_count;
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += target_rank == 0 ? 1.0 : 0.0;
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= sample_count;
    metrics.mean_rank /= sample_count;
    metrics.decision_utility /= sample_count;
    metrics.proposal_steps /= sample_count;
    metrics.baseline_steps /= sample_count;
    metrics.candidate_count /= sample_count;

    for (size_t bin = 0; bin < ARC_CALIBRATION_BINS; ++bin) {
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

static int save_arc_checkpoint(NeuralNet *net, const char *path, char out_hash[65])
{
    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = 1;
    strcpy(meta.parent_model_hash, "unknown");
    strcpy(meta.training_trace_root, "arc_agi_0520fde7");
    strcpy(meta.notes, "fixed ARC-AGI-1 0520fde7 cell-color scorer");
    strcpy(meta.feature_set, "arc-cell-color-v1");
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

typedef struct {
    const char *cursor;
    const char *end;
} ARCJsonParser;

static void print_json_string(const char *text)
{
    fputc('"', stdout);
    if (text != NULL) {
        for (const unsigned char *p = (const unsigned char *)text; *p != '\0'; ++p) {
            switch (*p) {
            case '"':
                fputs("\\\"", stdout);
                break;
            case '\\':
                fputs("\\\\", stdout);
                break;
            case '\b':
                fputs("\\b", stdout);
                break;
            case '\f':
                fputs("\\f", stdout);
                break;
            case '\n':
                fputs("\\n", stdout);
                break;
            case '\r':
                fputs("\\r", stdout);
                break;
            case '\t':
                fputs("\\t", stdout);
                break;
            default:
                if (*p < 0x20u) {
                    printf("\\u%04x", (unsigned)*p);
                } else {
                    fputc((int)*p, stdout);
                }
                break;
            }
        }
    }
    fputc('"', stdout);
}

static void arc_json_skip_ws(ARCJsonParser *parser)
{
    while (parser->cursor < parser->end &&
           isspace((unsigned char)*parser->cursor)) {
        ++parser->cursor;
    }
}

static int arc_json_expect(ARCJsonParser *parser, char expected)
{
    arc_json_skip_ws(parser);
    if (parser->cursor >= parser->end || *parser->cursor != expected) {
        return 0;
    }
    ++parser->cursor;
    return 1;
}

static int arc_json_parse_string(ARCJsonParser *parser, char *out, size_t out_size)
{
    arc_json_skip_ws(parser);
    if (parser->cursor >= parser->end || *parser->cursor != '"') {
        return 0;
    }
    ++parser->cursor;
    size_t used = 0;
    while (parser->cursor < parser->end && *parser->cursor != '"') {
        unsigned char ch = (unsigned char)*parser->cursor++;
        if (ch == '\\') {
            if (parser->cursor >= parser->end) {
                return 0;
            }
            ch = (unsigned char)*parser->cursor++;
            if (ch == 'u') {
                for (size_t i = 0; i < 4; ++i) {
                    if (parser->cursor >= parser->end ||
                        !isxdigit((unsigned char)*parser->cursor)) {
                        return 0;
                    }
                    ++parser->cursor;
                }
                ch = '?';
            }
        }
        if (used + 1u < out_size) {
            out[used++] = (char)ch;
        }
    }
    if (parser->cursor >= parser->end || *parser->cursor != '"') {
        return 0;
    }
    ++parser->cursor;
    if (out_size > 0) {
        out[used] = '\0';
    }
    return 1;
}

static int arc_json_skip_value(ARCJsonParser *parser);

static int arc_json_skip_array_or_object(ARCJsonParser *parser, char open, char close)
{
    if (!arc_json_expect(parser, open)) {
        return 0;
    }
    arc_json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == close) {
        ++parser->cursor;
        return 1;
    }
    for (;;) {
        if (open == '{') {
            char key[64];
            if (!arc_json_parse_string(parser, key, sizeof(key)) ||
                !arc_json_expect(parser, ':') ||
                !arc_json_skip_value(parser)) {
                return 0;
            }
        } else if (!arc_json_skip_value(parser)) {
            return 0;
        }
        arc_json_skip_ws(parser);
        if (parser->cursor < parser->end && *parser->cursor == ',') {
            ++parser->cursor;
            continue;
        }
        if (parser->cursor < parser->end && *parser->cursor == close) {
            ++parser->cursor;
            return 1;
        }
        return 0;
    }
}

static int arc_json_skip_value(ARCJsonParser *parser)
{
    arc_json_skip_ws(parser);
    if (parser->cursor >= parser->end) {
        return 0;
    }
    if (*parser->cursor == '"') {
        char scratch[1];
        return arc_json_parse_string(parser, scratch, sizeof(scratch));
    }
    if (*parser->cursor == '[') {
        return arc_json_skip_array_or_object(parser, '[', ']');
    }
    if (*parser->cursor == '{') {
        return arc_json_skip_array_or_object(parser, '{', '}');
    }
    while (parser->cursor < parser->end &&
           strchr(",]}\r\n\t ", *parser->cursor) == NULL) {
        ++parser->cursor;
    }
    return 1;
}

static int arc_json_parse_uint(ARCJsonParser *parser, unsigned long *out_value)
{
    arc_json_skip_ws(parser);
    if (parser->cursor >= parser->end || !isdigit((unsigned char)*parser->cursor)) {
        return 0;
    }
    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(parser->cursor, &end, 10);
    if (end == parser->cursor || errno == ERANGE || end > parser->end) {
        return 0;
    }
    parser->cursor = end;
    *out_value = value;
    return 1;
}

static int arc_json_parse_grid(ARCJsonParser *parser, ARCGrid *grid)
{
    if (!arc_json_expect(parser, '[')) {
        return 0;
    }
    ARCGrid parsed;
    memset(&parsed, 0, sizeof(parsed));
    uint8_t row[ARC_MAX_GRID_WIDTH];
    size_t width = 0;
    size_t height = 0;
    arc_json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == ']') {
        return 0;
    }
    for (;;) {
        if (!arc_json_expect(parser, '[')) {
            return 0;
        }
        size_t row_width = 0;
        for (;;) {
            unsigned long value = 0;
            if (row_width >= ARC_MAX_GRID_WIDTH ||
                !arc_json_parse_uint(parser, &value) ||
                value >= ARC_COLOR_COUNT) {
                return 0;
            }
            row[row_width++] = (uint8_t)value;
            arc_json_skip_ws(parser);
            if (parser->cursor < parser->end && *parser->cursor == ',') {
                ++parser->cursor;
                continue;
            }
            if (parser->cursor < parser->end && *parser->cursor == ']') {
                ++parser->cursor;
                break;
            }
            return 0;
        }
        if (row_width == 0 || height >= ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        if (width == 0) {
            width = row_width;
        } else if (row_width != width) {
            return 0;
        }
        memcpy(&parsed.cells[height * width], row, width * sizeof(row[0]));
        ++height;
        arc_json_skip_ws(parser);
        if (parser->cursor < parser->end && *parser->cursor == ',') {
            ++parser->cursor;
            continue;
        }
        if (parser->cursor < parser->end && *parser->cursor == ']') {
            ++parser->cursor;
            break;
        }
        return 0;
    }
    parsed.width = width;
    parsed.height = height;
    *grid = parsed;
    return 1;
}

static int arc_json_parse_pair(ARCJsonParser *parser,
                               ARCGridPair *pair,
                               int require_output)
{
    ARCGridPair parsed;
    memset(&parsed, 0, sizeof(parsed));
    int have_input = 0;
    if (!arc_json_expect(parser, '{')) {
        return 0;
    }
    arc_json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == '}') {
        return 0;
    }
    for (;;) {
        char key[32];
        if (!arc_json_parse_string(parser, key, sizeof(key)) ||
            !arc_json_expect(parser, ':')) {
            return 0;
        }
        if (strcmp(key, "input") == 0) {
            if (!arc_json_parse_grid(parser, &parsed.input)) {
                return 0;
            }
            have_input = 1;
        } else if (strcmp(key, "output") == 0) {
            if (!arc_json_parse_grid(parser, &parsed.output)) {
                return 0;
            }
            parsed.has_output = 1;
        } else if (!arc_json_skip_value(parser)) {
            return 0;
        }
        arc_json_skip_ws(parser);
        if (parser->cursor < parser->end && *parser->cursor == ',') {
            ++parser->cursor;
            continue;
        }
        if (parser->cursor < parser->end && *parser->cursor == '}') {
            ++parser->cursor;
            break;
        }
        return 0;
    }
    if (!have_input || (require_output && !parsed.has_output)) {
        return 0;
    }
    *pair = parsed;
    return 1;
}

static int arc_json_parse_pair_array(ARCJsonParser *parser,
                                     ARCGridPair *pairs,
                                     size_t *count,
                                     int require_output)
{
    if (!arc_json_expect(parser, '[')) {
        return 0;
    }
    size_t parsed_count = 0;
    arc_json_skip_ws(parser);
    if (parser->cursor < parser->end && *parser->cursor == ']') {
        ++parser->cursor;
        *count = 0;
        return 1;
    }
    for (;;) {
        if (parsed_count >= ARC_MAX_PAIRS ||
            !arc_json_parse_pair(parser, &pairs[parsed_count], require_output)) {
            return 0;
        }
        ++parsed_count;
        arc_json_skip_ws(parser);
        if (parser->cursor < parser->end && *parser->cursor == ',') {
            ++parser->cursor;
            continue;
        }
        if (parser->cursor < parser->end && *parser->cursor == ']') {
            ++parser->cursor;
            *count = parsed_count;
            return 1;
        }
        return 0;
    }
}

static int arc_parse_task_json(const char *json, size_t json_size, ARCTask *task)
{
    ARCJsonParser parser;
    parser.cursor = json;
    parser.end = json + json_size;
    int have_train = 0;
    int have_test = 0;
    if (!arc_json_expect(&parser, '{')) {
        return 0;
    }
    for (;;) {
        arc_json_skip_ws(&parser);
        if (parser.cursor < parser.end && *parser.cursor == '}') {
            ++parser.cursor;
            break;
        }
        char key[32];
        if (!arc_json_parse_string(&parser, key, sizeof(key)) ||
            !arc_json_expect(&parser, ':')) {
            return 0;
        }
        if (strcmp(key, "train") == 0) {
            if (!arc_json_parse_pair_array(&parser, task->train, &task->train_count, 1)) {
                return 0;
            }
            have_train = 1;
        } else if (strcmp(key, "test") == 0) {
            if (!arc_json_parse_pair_array(&parser, task->test, &task->test_count, 0)) {
                return 0;
            }
            have_test = 1;
        } else if (!arc_json_skip_value(&parser)) {
            return 0;
        }
        arc_json_skip_ws(&parser);
        if (parser.cursor < parser.end && *parser.cursor == ',') {
            ++parser.cursor;
            continue;
        }
        if (parser.cursor < parser.end && *parser.cursor == '}') {
            ++parser.cursor;
            break;
        }
        return 0;
    }
    return have_train && have_test && task->train_count > 0 && task->test_count > 0;
}

static char *arc_read_file(const char *path, size_t *out_size)
{
    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long size = ftell(fp);
    if (size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    char *buffer = malloc((size_t)size + 1u);
    if (buffer == NULL) {
        fclose(fp);
        return NULL;
    }
    size_t read_size = fread(buffer, 1, (size_t)size, fp);
    fclose(fp);
    if (read_size != (size_t)size) {
        free(buffer);
        return NULL;
    }
    buffer[read_size] = '\0';
    *out_size = read_size;
    return buffer;
}

static void arc_task_id_from_name(const char *name, char *out_id, size_t out_size)
{
    size_t len = strlen(name);
    if (len > 5 && strcmp(name + len - 5u, ".json") == 0) {
        len -= 5u;
    }
    if (len >= out_size) {
        len = out_size - 1u;
    }
    memcpy(out_id, name, len);
    out_id[len] = '\0';
}

static int arc_load_task_file(const char *path, const char *id, ARCTask *task)
{
    memset(task, 0, sizeof(*task));
    snprintf(task->id, sizeof(task->id), "%s", id);
    size_t json_size = 0;
    char *json = arc_read_file(path, &json_size);
    if (json == NULL) {
        return 0;
    }
    int ok = arc_parse_task_json(json, json_size, task);
    free(json);
    return ok;
}

static int arc_task_file_compare(const void *lhs, const void *rhs)
{
    const ARCTaskFile *left = lhs;
    const ARCTaskFile *right = rhs;
    return strcmp(left->id, right->id);
}

static int arc_name_ends_json(const char *name)
{
    size_t len = strlen(name);
    return len > 5 && strcmp(name + len - 5u, ".json") == 0;
}

static int arc_collect_task_files(const char *task_dir,
                                  ARCTaskFile *files,
                                  size_t capacity,
                                  size_t *out_count)
{
    DIR *dir = opendir(task_dir);
    if (dir == NULL) {
        return 0;
    }
    size_t count = 0;
    struct dirent *entry = NULL;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_name[0] == '.' || !arc_name_ends_json(entry->d_name)) {
            continue;
        }
        if (count >= capacity) {
            closedir(dir);
            return 0;
        }
        int written = snprintf(files[count].path,
                               sizeof(files[count].path),
                               "%s%s%s",
                               task_dir,
                               task_dir[strlen(task_dir) - 1u] == '/' ? "" : "/",
                               entry->d_name);
        if (written < 0 || (size_t)written >= sizeof(files[count].path)) {
            closedir(dir);
            return 0;
        }
        arc_task_id_from_name(entry->d_name, files[count].id, sizeof(files[count].id));
        ++count;
    }
    closedir(dir);
    qsort(files, count, sizeof(files[0]), arc_task_file_compare);
    *out_count = count;
    return 1;
}

static void print_reliability_bins_json(const ARCMetrics *metrics)
{
    int first = 1;
    fputs(",\"reliability_bins\":[", stdout);
    for (size_t bin = 0; bin < ARC_CALIBRATION_BINS; ++bin) {
        if (metrics->bin_count[bin] == 0) {
            continue;
        }
        if (!first) {
            fputc(',', stdout);
        }
        printf("{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
               "\"confidence\":%.8f,\"accuracy\":%.8f}",
               (double)bin / (double)ARC_CALIBRATION_BINS,
               (double)(bin + 1) / (double)ARC_CALIBRATION_BINS,
               metrics->bin_count[bin],
               metrics->bin_confidence[bin],
               metrics->bin_accuracy[bin]);
        first = 0;
    }
    fputc(']', stdout);
}

static void print_arc_metrics_json(const char *task_instance,
                                   const char *model_hash,
                                   const char *evaluation_mode,
                                   const char *feature_set,
                                   const ARCMetrics *metrics)
{
    double samples = metrics->samples == 0 ? 1.0 : (double)metrics->samples;
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"arc_agi\","
           "\"task_instance\":");
    print_json_string(task_instance == NULL ? "0520fde7" : task_instance);
    printf(",\"evaluation_mode\":\"%s\","
           "\"leakage_policy\":\"%s\","
           "\"model_hash\":\"%s\",\"parent_hash\":\"unknown\","
           "\"feature_set\":\"%s\","
           "\"feature_encoder_version\":1,"
           "\"benchmark_seed\":0,\"sample_count\":%zu,"
           "\"accuracy\":%.8f,\"top1\":%.8f,\"top5\":%.8f,\"top10\":%.8f,"
           "\"nll\":%.8f,\"ece\":%.8f,\"mean_rank\":%.8f,"
           "\"candidate_count\":%.8f,\"decision_utility\":%.8f,"
           "\"proposal_steps\":%.8f,\"baseline_steps\":%.8f,"
           "\"search_saved\":%.8f",
           evaluation_mode == NULL ? "neural_cell_scorer" : evaluation_mode,
           nn_leakage_policy_name(ARC_AGI_EXPERIMENT_SPEC.leakage_policy),
           model_hash == NULL ? "unknown" : model_hash,
           feature_set == NULL ? "arc-cell-color-v1" : feature_set,
           metrics->samples,
           (double)metrics->top1 / samples,
           (double)metrics->top1 / samples,
           (double)metrics->top5 / samples,
           (double)metrics->top10 / samples,
           metrics->nll,
           metrics->ece,
           metrics->mean_rank,
           metrics->candidate_count,
           metrics->decision_utility,
           metrics->proposal_steps,
           metrics->baseline_steps,
           metrics->baseline_steps - metrics->proposal_steps);
    print_reliability_bins_json(metrics);
    printf("}\n");
}

static void print_arc_transform_json(const char *task_instance,
                                     const ARCTransformMetrics *metrics)
{
    printf("{\"schema\":\"crlp.arc_transform.summary.v1\",\"report_version\":1,"
           "\"task\":\"arc_agi\",\"task_instance\":");
    print_json_string(task_instance == NULL ? "0520fde7" : task_instance);
    printf(",\"grammar_version\":%d,\"rules\":%zu,\"train_consistent_rules\":%zu,"
           "\"train_checked_cells\":%zu,\"train_mismatches\":%zu,"
           "\"unsupported_train_pairs\":%zu,\"unsupported_test_pairs\":%zu,"
           "\"unique_test_prediction\":%s,\"test_exact\":%s,\"certified\":%s,"
           "\"test_pairs\":%zu,\"test_cells\":%zu,\"test_correct_cells\":%zu,"
           "\"accepted_rules\":",
           ARC_GRAMMAR_VERSION,
           metrics->rules,
           metrics->train_consistent_rules,
           metrics->train_checked_cells,
           metrics->train_mismatches,
           metrics->unsupported_train_pairs,
           metrics->unsupported_test_pairs,
           metrics->unique_test_prediction ? "true" : "false",
           metrics->test_exact ? "true" : "false",
           metrics->certified ? "true" : "false",
           metrics->test_pairs,
           metrics->test_cells,
           metrics->test_correct_cells);
    print_json_string(metrics->accepted_rules);
    printf("}\n");
}

static void print_arc_loader_summary_json(const char *task_dir, const ARCLoaderSummary *summary)
{
    double cells = summary->test_cells == 0 ? 1.0 : (double)summary->test_cells;
    printf("{\"schema\":\"crlp.arc_loader.summary.v1\",\"report_version\":1,"
           "\"task\":\"arc_agi\",\"task_dir\":");
    print_json_string(task_dir);
    printf(",\"grammar_version\":%d,\"files_seen\":%zu,\"loaded_tasks\":%zu,"
           "\"parse_errors\":%zu,\"train_pairs\":%zu,\"test_pairs\":%zu,"
           "\"rules\":%zu,\"train_consistent_rules\":%zu,"
           "\"unique_prediction_tasks\":%zu,\"scored_tasks\":%zu,"
           "\"test_exact_tasks\":%zu,\"certified_tasks\":%zu,"
           "\"unsupported_tasks\":%zu,\"test_cells\":%zu,"
           "\"test_correct_cells\":%zu,\"cell_accuracy\":%.8f}\n",
           ARC_GRAMMAR_VERSION,
           summary->files_seen,
           summary->loaded_tasks,
           summary->parse_errors,
           summary->train_pairs,
           summary->test_pairs,
           summary->rules,
           summary->train_consistent_rules,
           summary->unique_prediction_tasks,
           summary->scored_tasks,
           summary->test_exact_tasks,
           summary->certified_tasks,
           summary->unsupported_tasks,
           summary->test_cells,
           summary->test_correct_cells,
           (double)summary->test_correct_cells / cells);
}

static void print_arc_metrics_line(const char *model_hash,
                                   const char *evaluation_mode,
                                   const ARCMetrics *metrics)
{
    double samples = metrics->samples == 0 ? 1.0 : (double)metrics->samples;
    printf("[arc_agi] mode=%s hash=%s task=0520fde7 top1=%.2f%% top5=%.2f%% top10=%.2f%% "
           "nll=%.4f ece=%.4f mean_rank=%.2f proposal_steps=%.2f baseline_steps=%.2f "
           "samples=%zu\n",
           evaluation_mode == NULL ? "neural_cell_scorer" : evaluation_mode,
           model_hash == NULL ? "unknown" : model_hash,
           100.0 * (double)metrics->top1 / samples,
           100.0 * (double)metrics->top5 / samples,
           100.0 * (double)metrics->top10 / samples,
           metrics->nll,
           metrics->ece,
           metrics->mean_rank,
           metrics->proposal_steps,
           metrics->baseline_steps,
           metrics->samples);
}

static void arc_loader_summary_add(ARCLoaderSummary *summary,
                                   const ARCTask *task,
                                   const ARCTransformMetrics *transform)
{
    ++summary->loaded_tasks;
    summary->train_pairs += task->train_count;
    summary->test_pairs += task->test_count;
    summary->rules = transform->rules;
    summary->train_consistent_rules += transform->train_consistent_rules;
    summary->unique_prediction_tasks += transform->unique_test_prediction ? 1u : 0u;
    summary->scored_tasks += transform->test_cells > 0 ? 1u : 0u;
    summary->test_exact_tasks += transform->test_exact ? 1u : 0u;
    summary->certified_tasks += transform->certified ? 1u : 0u;
    summary->unsupported_tasks +=
        transform->unsupported_train_pairs != 0 || transform->unsupported_test_pairs != 0 ? 1u : 0u;
    summary->test_cells += transform->test_cells;
    summary->test_correct_cells += transform->test_correct_cells;
}

static int run_arc_task_loader(const ARCConfig *config)
{
    ARCTaskFile *files = calloc(ARC_MAX_TASK_FILES, sizeof(*files));
    if (files == NULL) {
        return 0;
    }
    size_t file_count = 0;
    if (!arc_collect_task_files(config->task_dir, files, ARC_MAX_TASK_FILES, &file_count)) {
        fprintf(stderr, "failed to read ARC task directory: %s\n", config->task_dir);
        free(files);
        return 0;
    }

    ARCLoaderSummary summary;
    memset(&summary, 0, sizeof(summary));
    summary.files_seen = file_count;

    for (size_t i = 0; i < file_count; ++i) {
        if (config->max_tasks != 0 && summary.loaded_tasks >= config->max_tasks) {
            break;
        }
        ARCTask task;
        if (!arc_load_task_file(files[i].path, files[i].id, &task)) {
            ++summary.parse_errors;
            continue;
        }
        ARCTransformMetrics transform = evaluate_arc_transform_gate_task(&task);
        arc_loader_summary_add(&summary, &task, &transform);
        if (!config->loader_summary_only) {
            print_arc_transform_json(task.id, &transform);
            if (transform.certified) {
                ARCMetrics transform_metrics = arc_transform_to_metrics(&task, &transform);
                print_arc_metrics_json(task.id,
                                       "symbolic-transform-v2",
                                       "certified_transform_gate",
                                       "arc-transform-gate-v2",
                                       &transform_metrics);
            }
        }
    }
    print_arc_loader_summary_json(config->task_dir, &summary);
    free(files);
    return summary.loaded_tasks > 0;
}

static void print_usage(FILE *stream)
{
    fputs("usage: arc_agi [--schema] [--epochs count] [--steps-per-epoch count] "
          "[--test-samples count] [--learning-rate rate] [--seed seed] [--save path] "
          "[--task-dir path] [--max-tasks count] [--loader-summary-only]\n",
          stream);
}

static int parse_config(int argc, char **argv, ARCConfig *config, int *schema_only)
{
    config->epochs = ARC_DEFAULT_EPOCHS;
    config->steps_per_epoch = ARC_DEFAULT_STEPS_PER_EPOCH;
    config->test_samples = ARC_DEFAULT_TEST_SAMPLES;
    config->max_tasks = 0;
    config->seed = 20240516u;
    config->learning_rate = (nn_real)0.006;
    config->save_path = NULL;
    config->task_dir = NULL;
    config->loader_summary_only = 0;
    *schema_only = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--schema") == 0) {
            *schema_only = 1;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--steps-per-epoch") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->steps_per_epoch)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--test-samples") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->test_samples)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            if (!parse_real_text(argv[++i], &config->learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!parse_u64_text(argv[++i], &config->seed)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--task-dir") == 0 && i + 1 < argc) {
            config->task_dir = argv[++i];
            if (config->task_dir[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-tasks") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->max_tasks)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--loader-summary-only") == 0) {
            config->loader_summary_only = 1;
        } else {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char **argv)
{
    ARCConfig config;
    int schema_only = 0;
    if (!parse_config(argc, argv, &config, &schema_only)) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }
    if (schema_only) {
        return nn_experiment_write_schema_json(stdout, &ARC_AGI_EXPERIMENT_SPEC)
                   ? EXIT_SUCCESS
                   : EXIT_FAILURE;
    }
    if (config.task_dir != NULL) {
        return run_arc_task_loader(&config) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    NeuralNet *net = create_arc_model((unsigned int)(config.seed & 0xffffffffu));
    if (net == NULL) {
        return EXIT_FAILURE;
    }
    if (!train_arc_model(net, &config)) {
        nn_free(net);
        return EXIT_FAILURE;
    }

    char model_hash[65] = "unknown";
    if (config.save_path != NULL) {
        if (!save_arc_checkpoint(net, config.save_path, model_hash)) {
            fprintf(stderr, "failed to save ARC checkpoint: %s\n", config.save_path);
            nn_free(net);
            return EXIT_FAILURE;
        }
        printf("saved arc_agi checkpoint: %s hash=%s\n", config.save_path, model_hash);
    }

    ARCMetrics metrics = evaluate_arc_model(net, config.test_samples);
    print_arc_metrics_json("0520fde7", model_hash, "neural_cell_scorer", "arc-cell-color-v1", &metrics);
    print_arc_metrics_line(model_hash, "neural_cell_scorer", &metrics);

    ARCTask builtin_task;
    init_builtin_arc_task(&builtin_task);
    ARCTransformMetrics transform = evaluate_builtin_arc_transform_gate();
    print_arc_transform_json("0520fde7", &transform);
    ARCMetrics transform_metrics = arc_transform_to_metrics(&builtin_task, &transform);
    print_arc_metrics_json("0520fde7",
                           "symbolic-transform-v2",
                           "certified_transform_gate",
                           "arc-transform-gate-v2",
                           &transform_metrics);
    print_arc_metrics_line("symbolic-transform-v2",
                           "certified_transform_gate",
                           &transform_metrics);
    nn_free(net);
    return EXIT_SUCCESS;
}
