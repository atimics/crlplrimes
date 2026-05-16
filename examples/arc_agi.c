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
    ARC_ACCEPTED_RULES_BYTES = 2048,
    ARC_RULE_CANDIDATE_COUNT = 1024,
    ARC_GRAMMAR_VERSION = 4,
    ARC_PROGRAM_PROPOSER_BUDGET = 64,
    ARC_PROGRAM_BUDGET_GRID_COUNT = 4,
    ARC_PROGRAM_TASK_FEATURE_COUNT = 24,
    ARC_PROGRAM_FEATURE_COUNT = NN_PROGRAM_FEATURE_COUNT + ARC_PROGRAM_TASK_FEATURE_COUNT,
    ARC_PROGRAM_TRAIN_EPOCHS = 5,
    ARC_PROGRAM_STRATEGY_COUNT = 5,
};

static const size_t ARC_PROGRAM_BUDGET_GRID[ARC_PROGRAM_BUDGET_GRID_COUNT] = {
    8,
    16,
    32,
    64,
};

typedef enum {
    ARC_RULE_COPY_LEFT = 0,
    ARC_RULE_COPY_RIGHT = 1,
    ARC_RULE_INTERSECTION = 2,
    ARC_RULE_UNION = 3,
    ARC_RULE_OBJECT_SELECT = 4,
    ARC_RULE_COLOR_MAP_LEFT = 5,
    ARC_RULE_GRID_IDENTITY = 6,
    ARC_RULE_GRID_COLOR_MAP = 7,
    ARC_RULE_GRID_FOREGROUND_RECOLOR = 8,
    ARC_RULE_CROP_BACKGROUND = 9,
    ARC_RULE_CROP_BACKGROUND_RECOLOR = 10,
    ARC_RULE_CROP_COLOR = 11,
    ARC_RULE_CROP_COLOR_MASK = 12,
    ARC_RULE_ROTATE90 = 13,
    ARC_RULE_ROTATE180 = 14,
    ARC_RULE_ROTATE270 = 15,
    ARC_RULE_FLIP_HORIZONTAL = 16,
    ARC_RULE_FLIP_VERTICAL = 17,
    ARC_RULE_TRANSPOSE = 18,
    ARC_RULE_SCALE = 19,
    ARC_RULE_BBOX_OUTLINE = 20,
    ARC_RULE_BBOX_FILL = 21,
    ARC_RULE_TRANSLATE = 22,
    ARC_RULE_OBJECT_CROP_LARGEST = 23,
    ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR = 24,
    ARC_RULE_OBJECT_MASK_LARGEST = 25,
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
    .rule_grammar =
        "copy_left, copy_right, mask_intersection, mask_union, object_select_overlap, "
        "color_map, same_grid_color_map, crop, rotate, flip, transpose, scale, bbox, translate, "
        "connected_component_largest",
    .certificate_checker =
        "train-pair consistency plus unique grammar-consistent test prediction, reported against compiled test output",
    .gate_executor = "certified transform gate emits exact color candidates when the grammar prediction is unique",
    .proposer_inputs =
        "typed NNProgram transform descriptors encoded by nn_program_encode_features, "
        "plus ARC train-pair shape/color/object descriptors",
    .proposer_targets =
        "train-split exact transform success for learned ranking; held-out exact-cell agreement remains diagnostic",
    .proposer_mode = NN_PROPOSER_SELECTION,
    .baseline_orderings =
        "neural scorer, majority color, certified transform gate, program proposer orderings",
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
    size_t program_budget;
    size_t program_train_max_tasks;
    uint64_t seed;
    nn_real learning_rate;
    nn_real program_learning_rate;
    const char *save_path;
    const char *task_dir;
    const char *program_train_dir;
    int loader_summary_only;
    int program_budget_grid;
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
    size_t count;
    size_t min_x;
    size_t min_y;
    size_t max_x;
    size_t max_y;
    uint8_t mask[ARC_MAX_GRID_CELLS];
} ARCComponent;

typedef struct {
    ARCTransformRule op;
    uint8_t input_color;
    uint8_t output_color;
    uint8_t background_color;
    uint8_t color_map[ARC_COLOR_COUNT];
    int dx;
    int dy;
    size_t scale;
    char name[96];
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
    int train_consistent;
    int test_supported;
    uint64_t prediction_hash;
    size_t train_checked_cells;
    size_t train_mismatches;
    size_t unsupported_train_pairs;
    size_t unsupported_test_pairs;
    size_t test_cells;
    size_t test_correct_cells;
    uint64_t program_hash;
    int object_program;
} ARCProgramEval;

typedef enum {
    ARC_PROGRAM_STRATEGY_GRAMMAR = 0,
    ARC_PROGRAM_STRATEGY_STRUCTURAL = 1,
    ARC_PROGRAM_STRATEGY_RANDOM = 2,
    ARC_PROGRAM_STRATEGY_LEARNED = 3,
    ARC_PROGRAM_STRATEGY_ORACLE = 4,
} ARCProgramStrategy;

typedef struct {
    size_t candidate_index;
    double primary;
    double secondary;
    uint64_t tie_breaker;
} ARCProgramRank;

typedef struct {
    NNProgramProposerSummary summary;
    uint64_t program_set_hash;
    char program_set_hash_text[32];
    size_t unique_prediction_tasks;
    size_t exact_tasks;
} ARCProgramProposerAggregate;

typedef struct {
    NeuralNet *net;
    size_t trained_tasks;
    size_t train_updates;
    double family_score[32];
} ARCProgramLearnedState;

typedef struct {
    double input_width;
    double input_height;
    double output_width;
    double output_height;
    double width_ratio;
    double height_ratio;
    double area_ratio;
    double same_shape_fraction;
    double output_smaller_fraction;
    double output_larger_fraction;
    double input_color_count;
    double output_color_count;
    double input_foreground_fraction;
    double output_foreground_fraction;
    double input_component_count;
    double output_component_count;
    double input_bbox_width_fraction;
    double input_bbox_height_fraction;
    double output_bbox_width_fraction;
    double output_bbox_height_fraction;
    double bbox_area_ratio;
    double component_delta;
    double color_delta;
    double foreground_delta;
} ARCProgramTaskFeatures;

_Static_assert(ARC_PROGRAM_TASK_FEATURE_COUNT == 24,
               "ARC program task feature writer must match the feature count");

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

static void arc_init_candidate(ARCTransformCandidate *candidate, ARCTransformRule op, const char *name)
{
    memset(candidate, 0, sizeof(*candidate));
    candidate->op = op;
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        candidate->color_map[color] = color;
    }
    snprintf(candidate->name, sizeof(candidate->name), "%s", name);
}

static int arc_add_candidate(ARCTransformCandidate *candidates,
                             size_t capacity,
                             size_t *count,
                             const ARCTransformCandidate *candidate)
{
    if (*count >= capacity) {
        return 0;
    }
    candidates[*count] = *candidate;
    ++*count;
    return 1;
}

static int arc_add_named_candidate(ARCTransformCandidate *candidates,
                                   size_t capacity,
                                   size_t *count,
                                   ARCTransformRule op,
                                   const char *name)
{
    ARCTransformCandidate candidate;
    arc_init_candidate(&candidate, op, name);
    return arc_add_candidate(candidates, capacity, count, &candidate);
}

static int arc_build_observed_color_map_candidate(const ARCTask *task,
                                                  ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_GRID_COLOR_MAP, "grid_color_map_same_shape");
    int seen[ARC_COLOR_COUNT];
    memset(seen, 0, sizeof(seen));
    int mapped_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *output = &task->train[pair].output;
        if (!task->train[pair].has_output || !arc_grid_same_shape(input, output)) {
            return 0;
        }
        size_t cells = arc_grid_cells(input);
        for (size_t cell = 0; cell < cells; ++cell) {
            uint8_t in = input->cells[cell];
            uint8_t out = output->cells[cell];
            if (!seen[in]) {
                candidate->color_map[in] = out;
                seen[in] = 1;
                mapped_any = 1;
            } else if (candidate->color_map[in] != out) {
                return 0;
            }
        }
    }
    return mapped_any;
}

static size_t arc_build_candidates(const ARCTask *task,
                                   ARCTransformCandidate *candidates,
                                   size_t capacity)
{
    size_t count = 0;
    ARCTransformCandidate candidate;
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
            arc_init_candidate(&candidate, colored_templates[template_id].op, "");
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "%s%u",
                     colored_templates[template_id].prefix,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_COLOR_MAP_LEFT, "color_map_left_identity");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_GRID_IDENTITY, "grid_identity");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_ROTATE90, "rotate90");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_ROTATE180, "rotate180");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_ROTATE270, "rotate270");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_FLIP_HORIZONTAL, "flip_horizontal");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_FLIP_VERTICAL, "flip_vertical");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_TRANSPOSE, "transpose");

    for (size_t scale = 2; scale <= 4; ++scale) {
        arc_init_candidate(&candidate, ARC_RULE_SCALE, "");
        candidate.scale = scale;
        snprintf(candidate.name, sizeof(candidate.name), "scale_%zux", scale);
        arc_add_candidate(candidates, capacity, &count, &candidate);
    }

    if (arc_build_observed_color_map_candidate(task, &candidate)) {
        arc_add_candidate(candidates, capacity, &count, &candidate);
    }

    for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
        arc_init_candidate(&candidate, ARC_RULE_CROP_BACKGROUND, "");
        candidate.background_color = background;
        snprintf(candidate.name, sizeof(candidate.name), "crop_non_background_%u", (unsigned)background);
        arc_add_candidate(candidates, capacity, &count, &candidate);

        for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
            arc_init_candidate(&candidate, ARC_RULE_GRID_FOREGROUND_RECOLOR, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "grid_recolor_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_CROP_BACKGROUND_RECOLOR, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "crop_recolor_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_BBOX_OUTLINE, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "bbox_outline_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_BBOX_FILL, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "bbox_fill_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
        arc_init_candidate(&candidate, ARC_RULE_OBJECT_CROP_LARGEST, "");
        candidate.background_color = background;
        snprintf(candidate.name,
                 sizeof(candidate.name),
                 "object_crop_largest_non_background_%u",
                 (unsigned)background);
        arc_add_candidate(candidates, capacity, &count, &candidate);

        for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
            arc_init_candidate(&candidate, ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "object_crop_largest_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_OBJECT_MASK_LARGEST, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "object_mask_largest_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    for (uint8_t input_color = 1; input_color < ARC_COLOR_COUNT; ++input_color) {
        arc_init_candidate(&candidate, ARC_RULE_CROP_COLOR, "");
        candidate.input_color = input_color;
        snprintf(candidate.name, sizeof(candidate.name), "crop_color_%u", (unsigned)input_color);
        arc_add_candidate(candidates, capacity, &count, &candidate);

        for (uint8_t output_color = 1; output_color < ARC_COLOR_COUNT; ++output_color) {
            arc_init_candidate(&candidate, ARC_RULE_CROP_COLOR_MASK, "");
            candidate.input_color = input_color;
            candidate.output_color = output_color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "crop_color_%u_mask_to_%u",
                     (unsigned)input_color,
                     (unsigned)output_color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    static const int shifts[][2] = {
        {-2, 0}, {-1, 0}, {1, 0}, {2, 0}, {0, -2}, {0, -1}, {0, 1}, {0, 2},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
    };
    for (size_t shift = 0; shift < sizeof(shifts) / sizeof(shifts[0]); ++shift) {
        arc_init_candidate(&candidate, ARC_RULE_TRANSLATE, "");
        candidate.background_color = 0;
        candidate.dx = shifts[shift][0];
        candidate.dy = shifts[shift][1];
        snprintf(candidate.name,
                 sizeof(candidate.name),
                 "translate_nonzero_%+d_%+d",
                 candidate.dx,
                 candidate.dy);
        arc_add_candidate(candidates, capacity, &count, &candidate);
    }
    return count;
}

static int arc_find_non_background_bbox(const ARCGrid *grid,
                                        uint8_t background,
                                        size_t *min_x,
                                        size_t *min_y,
                                        size_t *max_x,
                                        size_t *max_y)
{
    int found = 0;
    *min_x = grid->width;
    *min_y = grid->height;
    *max_x = 0;
    *max_y = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        for (size_t x = 0; x < grid->width; ++x) {
            if (grid->cells[y * grid->width + x] == background) {
                continue;
            }
            if (!found || x < *min_x) {
                *min_x = x;
            }
            if (!found || y < *min_y) {
                *min_y = y;
            }
            if (!found || x > *max_x) {
                *max_x = x;
            }
            if (!found || y > *max_y) {
                *max_y = y;
            }
            found = 1;
        }
    }
    return found;
}

static int arc_find_color_bbox(const ARCGrid *grid,
                               uint8_t color,
                               size_t *min_x,
                               size_t *min_y,
                               size_t *max_x,
                               size_t *max_y)
{
    int found = 0;
    *min_x = grid->width;
    *min_y = grid->height;
    *max_x = 0;
    *max_y = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        for (size_t x = 0; x < grid->width; ++x) {
            if (grid->cells[y * grid->width + x] != color) {
                continue;
            }
            if (!found || x < *min_x) {
                *min_x = x;
            }
            if (!found || y < *min_y) {
                *min_y = y;
            }
            if (!found || x > *max_x) {
                *max_x = x;
            }
            if (!found || y > *max_y) {
                *max_y = y;
            }
            found = 1;
        }
    }
    return found;
}

static void arc_component_clear(ARCComponent *component)
{
    memset(component, 0, sizeof(*component));
    component->min_x = ARC_MAX_GRID_WIDTH;
    component->min_y = ARC_MAX_GRID_HEIGHT;
}

static int arc_find_largest_component(const ARCGrid *grid,
                                      uint8_t background,
                                      ARCComponent *component)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    uint8_t current_mask[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    memset(visited, 0, sizeof(visited));
    arc_component_clear(component);

    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        if (visited[start] || grid->cells[start] == background) {
            continue;
        }
        memset(current_mask, 0, sizeof(current_mask));
        size_t head = 0;
        size_t tail = 0;
        size_t count = 0;
        size_t min_x = grid->width;
        size_t min_y = grid->height;
        size_t max_x = 0;
        size_t max_y = 0;

        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            current_mask[cell] = 1;
            ++count;
            if (x < min_x) {
                min_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y > max_y) {
                max_y = y;
            }

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (size_t i = 0; i < 4; ++i) {
                int nx = (int)x + offsets[i][0];
                int ny = (int)y + offsets[i][1];
                if (nx < 0 || ny < 0 ||
                    nx >= (int)grid->width || ny >= (int)grid->height) {
                    continue;
                }
                size_t next = (size_t)ny * grid->width + (size_t)nx;
                if (visited[next] || grid->cells[next] == background) {
                    continue;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }

        int better = count > component->count ||
                     (count == component->count && component->count > 0 &&
                      (min_y < component->min_y ||
                       (min_y == component->min_y && min_x < component->min_x)));
        if (better) {
            component->count = count;
            component->min_x = min_x;
            component->min_y = min_y;
            component->max_x = max_x;
            component->max_y = max_y;
            memcpy(component->mask, current_mask, sizeof(component->mask));
        }
    }
    return component->count > 0;
}

static size_t arc_count_components(const ARCGrid *grid, uint8_t background)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    size_t components = 0;
    memset(visited, 0, sizeof(visited));

    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        if (visited[start] || grid->cells[start] == background) {
            continue;
        }
        ++components;
        size_t head = 0;
        size_t tail = 0;
        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (size_t i = 0; i < 4; ++i) {
                int nx = (int)x + offsets[i][0];
                int ny = (int)y + offsets[i][1];
                if (nx < 0 || ny < 0 ||
                    nx >= (int)grid->width || ny >= (int)grid->height) {
                    continue;
                }
                size_t next = (size_t)ny * grid->width + (size_t)nx;
                if (visited[next] || grid->cells[next] == background) {
                    continue;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }
    }
    return components;
}

static size_t arc_count_colors(const ARCGrid *grid)
{
    uint8_t seen[ARC_COLOR_COUNT];
    memset(seen, 0, sizeof(seen));
    size_t cells = arc_grid_cells(grid);
    size_t count = 0;
    for (size_t cell = 0; cell < cells; ++cell) {
        uint8_t color = grid->cells[cell];
        if (color < ARC_COLOR_COUNT && !seen[color]) {
            seen[color] = 1;
            ++count;
        }
    }
    return count;
}

static size_t arc_count_non_background(const ARCGrid *grid, uint8_t background)
{
    size_t cells = arc_grid_cells(grid);
    size_t count = 0;
    for (size_t cell = 0; cell < cells; ++cell) {
        count += grid->cells[cell] != background ? 1u : 0u;
    }
    return count;
}

static double arc_safe_ratio(double numerator, double denominator)
{
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

static double arc_clamp_signed_unit(double value)
{
    if (value > 1.0) {
        return 1.0;
    }
    if (value < -1.0) {
        return -1.0;
    }
    return value;
}

static void arc_program_task_features_add_grid_pair(ARCProgramTaskFeatures *features,
                                                    const ARCGrid *input,
                                                    const ARCGrid *output)
{
    double input_cells = (double)arc_grid_cells(input);
    double output_cells = (double)arc_grid_cells(output);
    size_t in_min_x = 0;
    size_t in_min_y = 0;
    size_t in_max_x = 0;
    size_t in_max_y = 0;
    size_t out_min_x = 0;
    size_t out_min_y = 0;
    size_t out_max_x = 0;
    size_t out_max_y = 0;
    int in_bbox = arc_find_non_background_bbox(input, 0, &in_min_x, &in_min_y, &in_max_x, &in_max_y);
    int out_bbox =
        arc_find_non_background_bbox(output, 0, &out_min_x, &out_min_y, &out_max_x, &out_max_y);
    double in_bbox_width = in_bbox ? (double)(in_max_x - in_min_x + 1u) : 0.0;
    double in_bbox_height = in_bbox ? (double)(in_max_y - in_min_y + 1u) : 0.0;
    double out_bbox_width = out_bbox ? (double)(out_max_x - out_min_x + 1u) : 0.0;
    double out_bbox_height = out_bbox ? (double)(out_max_y - out_min_y + 1u) : 0.0;

    features->input_width += (double)input->width / (double)ARC_MAX_GRID_WIDTH;
    features->input_height += (double)input->height / (double)ARC_MAX_GRID_HEIGHT;
    features->output_width += (double)output->width / (double)ARC_MAX_GRID_WIDTH;
    features->output_height += (double)output->height / (double)ARC_MAX_GRID_HEIGHT;
    features->width_ratio += arc_safe_ratio((double)output->width, (double)input->width);
    features->height_ratio += arc_safe_ratio((double)output->height, (double)input->height);
    features->area_ratio += arc_safe_ratio(output_cells, input_cells);
    features->same_shape_fraction +=
        input->width == output->width && input->height == output->height ? 1.0 : 0.0;
    features->output_smaller_fraction += output_cells < input_cells ? 1.0 : 0.0;
    features->output_larger_fraction += output_cells > input_cells ? 1.0 : 0.0;
    features->input_color_count += (double)arc_count_colors(input) / (double)ARC_COLOR_COUNT;
    features->output_color_count += (double)arc_count_colors(output) / (double)ARC_COLOR_COUNT;
    features->input_foreground_fraction +=
        arc_safe_ratio((double)arc_count_non_background(input, 0), input_cells);
    features->output_foreground_fraction +=
        arc_safe_ratio((double)arc_count_non_background(output, 0), output_cells);
    features->input_component_count +=
        (double)arc_count_components(input, 0) / (double)ARC_MAX_GRID_CELLS;
    features->output_component_count +=
        (double)arc_count_components(output, 0) / (double)ARC_MAX_GRID_CELLS;
    features->input_bbox_width_fraction += arc_safe_ratio(in_bbox_width, (double)input->width);
    features->input_bbox_height_fraction += arc_safe_ratio(in_bbox_height, (double)input->height);
    features->output_bbox_width_fraction += arc_safe_ratio(out_bbox_width, (double)output->width);
    features->output_bbox_height_fraction += arc_safe_ratio(out_bbox_height, (double)output->height);
    features->bbox_area_ratio +=
        arc_safe_ratio(out_bbox_width * out_bbox_height, in_bbox_width * in_bbox_height);
    features->component_delta +=
        ((double)arc_count_components(output, 0) - (double)arc_count_components(input, 0)) /
        (double)ARC_MAX_GRID_CELLS;
    features->color_delta +=
        ((double)arc_count_colors(output) - (double)arc_count_colors(input)) /
        (double)ARC_COLOR_COUNT;
    features->foreground_delta +=
        arc_safe_ratio((double)arc_count_non_background(output, 0), output_cells) -
        arc_safe_ratio((double)arc_count_non_background(input, 0), input_cells);
}

static void arc_program_task_features_normalize(ARCProgramTaskFeatures *features, double count)
{
    if (count <= 0.0) {
        return;
    }
#define ARC_NORMALIZE_TASK_FEATURE(field) \
    features->field = arc_clamp_signed_unit(features->field / count)
    ARC_NORMALIZE_TASK_FEATURE(input_width);
    ARC_NORMALIZE_TASK_FEATURE(input_height);
    ARC_NORMALIZE_TASK_FEATURE(output_width);
    ARC_NORMALIZE_TASK_FEATURE(output_height);
    ARC_NORMALIZE_TASK_FEATURE(width_ratio);
    ARC_NORMALIZE_TASK_FEATURE(height_ratio);
    ARC_NORMALIZE_TASK_FEATURE(area_ratio);
    ARC_NORMALIZE_TASK_FEATURE(same_shape_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_smaller_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_larger_fraction);
    ARC_NORMALIZE_TASK_FEATURE(input_color_count);
    ARC_NORMALIZE_TASK_FEATURE(output_color_count);
    ARC_NORMALIZE_TASK_FEATURE(input_foreground_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_foreground_fraction);
    ARC_NORMALIZE_TASK_FEATURE(input_component_count);
    ARC_NORMALIZE_TASK_FEATURE(output_component_count);
    ARC_NORMALIZE_TASK_FEATURE(input_bbox_width_fraction);
    ARC_NORMALIZE_TASK_FEATURE(input_bbox_height_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_bbox_width_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_bbox_height_fraction);
    ARC_NORMALIZE_TASK_FEATURE(bbox_area_ratio);
    ARC_NORMALIZE_TASK_FEATURE(component_delta);
    ARC_NORMALIZE_TASK_FEATURE(color_delta);
    ARC_NORMALIZE_TASK_FEATURE(foreground_delta);
#undef ARC_NORMALIZE_TASK_FEATURE
}

static void arc_program_task_features(const ARCTask *task, ARCProgramTaskFeatures *features)
{
    memset(features, 0, sizeof(*features));
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        arc_program_task_features_add_grid_pair(features,
                                                &task->train[pair].input,
                                                &task->train[pair].output);
    }
    arc_program_task_features_normalize(features, (double)task->train_count);
}

static void arc_program_task_features_write(const ARCProgramTaskFeatures *features,
                                            nn_real *out)
{
    out[0] = (nn_real)features->input_width;
    out[1] = (nn_real)features->input_height;
    out[2] = (nn_real)features->output_width;
    out[3] = (nn_real)features->output_height;
    out[4] = (nn_real)features->width_ratio;
    out[5] = (nn_real)features->height_ratio;
    out[6] = (nn_real)features->area_ratio;
    out[7] = (nn_real)features->same_shape_fraction;
    out[8] = (nn_real)features->output_smaller_fraction;
    out[9] = (nn_real)features->output_larger_fraction;
    out[10] = (nn_real)features->input_color_count;
    out[11] = (nn_real)features->output_color_count;
    out[12] = (nn_real)features->input_foreground_fraction;
    out[13] = (nn_real)features->output_foreground_fraction;
    out[14] = (nn_real)features->input_component_count;
    out[15] = (nn_real)features->output_component_count;
    out[16] = (nn_real)features->input_bbox_width_fraction;
    out[17] = (nn_real)features->input_bbox_height_fraction;
    out[18] = (nn_real)features->output_bbox_width_fraction;
    out[19] = (nn_real)features->output_bbox_height_fraction;
    out[20] = (nn_real)features->bbox_area_ratio;
    out[21] = (nn_real)features->component_delta;
    out[22] = (nn_real)features->color_delta;
    out[23] = (nn_real)features->foreground_delta;
}

static int arc_apply_largest_component_candidate(const ARCTransformCandidate *candidate,
                                                 const ARCGrid *input,
                                                 ARCGrid *output)
{
    ARCComponent component;
    if (!arc_find_largest_component(input, candidate->background_color, &component)) {
        return 0;
    }

    if (candidate->op == ARC_RULE_OBJECT_MASK_LARGEST) {
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
            output->cells[cell] = component.mask[cell] ? candidate->output_color : 0;
        }
        return 1;
    }

    size_t width = component.max_x - component.min_x + 1u;
    size_t height = component.max_y - component.min_y + 1u;
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t source = (component.min_y + y) * input->width + component.min_x + x;
            size_t target = y * width + x;
            if (!component.mask[source]) {
                output->cells[target] = 0;
            } else if (candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR) {
                output->cells[target] = candidate->output_color;
            } else {
                output->cells[target] = input->cells[source];
            }
        }
    }
    return 1;
}

static int arc_apply_split_panel_candidate(const ARCTransformCandidate *candidate,
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

static int arc_apply_crop_background(const ARCTransformCandidate *candidate,
                                     const ARCGrid *input,
                                     ARCGrid *output,
                                     int recolor)
{
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    if (!arc_find_non_background_bbox(input,
                                      candidate->background_color,
                                      &min_x,
                                      &min_y,
                                      &max_x,
                                      &max_y)) {
        return 0;
    }
    size_t width = max_x - min_x + 1u;
    size_t height = max_y - min_y + 1u;
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            uint8_t value = input->cells[(min_y + y) * input->width + min_x + x];
            if (recolor) {
                value = value == candidate->background_color ? 0 : candidate->output_color;
            }
            output->cells[y * width + x] = value;
        }
    }
    return 1;
}

static int arc_apply_crop_color(const ARCTransformCandidate *candidate,
                                const ARCGrid *input,
                                ARCGrid *output,
                                int mask_only)
{
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    if (!arc_find_color_bbox(input,
                             candidate->input_color,
                             &min_x,
                             &min_y,
                             &max_x,
                             &max_y)) {
        return 0;
    }
    size_t width = max_x - min_x + 1u;
    size_t height = max_y - min_y + 1u;
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            uint8_t value = input->cells[(min_y + y) * input->width + min_x + x];
            if (mask_only) {
                value = value == candidate->input_color ? candidate->output_color : 0;
            }
            output->cells[y * width + x] = value;
        }
    }
    return 1;
}

static int arc_apply_candidate(const ARCTransformCandidate *candidate,
                               const ARCGrid *input,
                               ARCGrid *output)
{
    switch (candidate->op) {
    case ARC_RULE_COPY_LEFT:
    case ARC_RULE_COPY_RIGHT:
    case ARC_RULE_INTERSECTION:
    case ARC_RULE_UNION:
    case ARC_RULE_OBJECT_SELECT:
    case ARC_RULE_COLOR_MAP_LEFT:
        return arc_apply_split_panel_candidate(candidate, input, output);
    case ARC_RULE_GRID_IDENTITY:
        *output = *input;
        return 1;
    case ARC_RULE_GRID_COLOR_MAP:
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
            output->cells[cell] = candidate->color_map[input->cells[cell]];
        }
        return 1;
    case ARC_RULE_GRID_FOREGROUND_RECOLOR:
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
            output->cells[cell] =
                input->cells[cell] == candidate->background_color ? 0 : candidate->output_color;
        }
        return 1;
    case ARC_RULE_CROP_BACKGROUND:
        return arc_apply_crop_background(candidate, input, output, 0);
    case ARC_RULE_CROP_BACKGROUND_RECOLOR:
        return arc_apply_crop_background(candidate, input, output, 1);
    case ARC_RULE_CROP_COLOR:
        return arc_apply_crop_color(candidate, input, output, 0);
    case ARC_RULE_CROP_COLOR_MASK:
        return arc_apply_crop_color(candidate, input, output, 1);
    case ARC_RULE_OBJECT_CROP_LARGEST:
    case ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR:
    case ARC_RULE_OBJECT_MASK_LARGEST:
        return arc_apply_largest_component_candidate(candidate, input, output);
    case ARC_RULE_ROTATE90:
        if (input->height > ARC_MAX_GRID_WIDTH || input->width > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->height, input->width);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                size_t out_x = input->height - 1u - y;
                size_t out_y = x;
                output->cells[out_y * output->width + out_x] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_ROTATE180:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                size_t out_x = input->width - 1u - x;
                size_t out_y = input->height - 1u - y;
                output->cells[out_y * output->width + out_x] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_ROTATE270:
        if (input->height > ARC_MAX_GRID_WIDTH || input->width > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->height, input->width);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                size_t out_x = y;
                size_t out_y = input->width - 1u - x;
                output->cells[out_y * output->width + out_x] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_FLIP_HORIZONTAL:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                output->cells[y * output->width + (input->width - 1u - x)] =
                    input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_FLIP_VERTICAL:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                output->cells[(input->height - 1u - y) * output->width + x] =
                    input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_TRANSPOSE:
        if (input->height > ARC_MAX_GRID_WIDTH || input->width > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->height, input->width);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                output->cells[x * output->width + y] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_SCALE:
        if (candidate->scale == 0 ||
            input->width * candidate->scale > ARC_MAX_GRID_WIDTH ||
            input->height * candidate->scale > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->width * candidate->scale, input->height * candidate->scale);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                uint8_t value = input->cells[y * input->width + x];
                for (size_t yy = 0; yy < candidate->scale; ++yy) {
                    for (size_t xx = 0; xx < candidate->scale; ++xx) {
                        size_t out_x = x * candidate->scale + xx;
                        size_t out_y = y * candidate->scale + yy;
                        output->cells[out_y * output->width + out_x] = value;
                    }
                }
            }
        }
        return 1;
    case ARC_RULE_BBOX_OUTLINE:
    case ARC_RULE_BBOX_FILL: {
        size_t min_x = 0;
        size_t min_y = 0;
        size_t max_x = 0;
        size_t max_y = 0;
        if (!arc_find_non_background_bbox(input,
                                          candidate->background_color,
                                          &min_x,
                                          &min_y,
                                          &max_x,
                                          &max_y)) {
            return 0;
        }
        arc_init_grid(output, input->width, input->height);
        for (size_t y = min_y; y <= max_y; ++y) {
            for (size_t x = min_x; x <= max_x; ++x) {
                if (candidate->op == ARC_RULE_BBOX_FILL ||
                    x == min_x || x == max_x || y == min_y || y == max_y) {
                    output->cells[y * output->width + x] = candidate->output_color;
                }
            }
        }
        return 1;
    }
    case ARC_RULE_TRANSLATE:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                uint8_t value = input->cells[y * input->width + x];
                if (value == candidate->background_color) {
                    continue;
                }
                int out_x = (int)x + candidate->dx;
                int out_y = (int)y + candidate->dy;
                if (out_x < 0 || out_y < 0 ||
                    out_x >= (int)input->width || out_y >= (int)input->height) {
                    return 0;
                }
                output->cells[(size_t)out_y * output->width + (size_t)out_x] = value;
            }
        }
        return 1;
    default:
        return 0;
    }
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

static uint64_t arc_hash_mix(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}

static size_t arc_task_test_cells(const ARCTask *task)
{
    size_t cells = 0;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        if (task->test[pair].has_output) {
            cells += arc_grid_cells(&task->test[pair].output);
        }
    }
    return cells;
}

static uint64_t arc_prediction_hash(const ARCGrid *predictions, size_t count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t pair = 0; pair < count; ++pair) {
        hash = arc_hash_mix(hash, predictions[pair].width);
        hash = arc_hash_mix(hash, predictions[pair].height);
        size_t cells = arc_grid_cells(&predictions[pair]);
        for (size_t cell = 0; cell < cells; ++cell) {
            hash = arc_hash_mix(hash, predictions[pair].cells[cell]);
        }
    }
    return hash;
}

static int arc_candidate_is_object_program(const ARCTransformCandidate *candidate)
{
    return candidate->op == ARC_RULE_OBJECT_CROP_LARGEST ||
           candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR ||
           candidate->op == ARC_RULE_OBJECT_MASK_LARGEST;
}

static size_t arc_candidate_family(const ARCTransformCandidate *candidate)
{
    switch (candidate->op) {
    case ARC_RULE_GRID_COLOR_MAP:
    case ARC_RULE_COLOR_MAP_LEFT:
        return 1;
    case ARC_RULE_ROTATE90:
    case ARC_RULE_ROTATE180:
    case ARC_RULE_ROTATE270:
    case ARC_RULE_FLIP_HORIZONTAL:
    case ARC_RULE_FLIP_VERTICAL:
    case ARC_RULE_TRANSPOSE:
        return 2;
    case ARC_RULE_SCALE:
        return 3;
    case ARC_RULE_CROP_BACKGROUND:
    case ARC_RULE_CROP_BACKGROUND_RECOLOR:
    case ARC_RULE_CROP_COLOR:
    case ARC_RULE_CROP_COLOR_MASK:
        return 4;
    case ARC_RULE_OBJECT_CROP_LARGEST:
    case ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR:
    case ARC_RULE_OBJECT_MASK_LARGEST:
        return 5;
    case ARC_RULE_COPY_LEFT:
    case ARC_RULE_COPY_RIGHT:
    case ARC_RULE_INTERSECTION:
    case ARC_RULE_UNION:
    case ARC_RULE_OBJECT_SELECT:
        return 6;
    case ARC_RULE_BBOX_OUTLINE:
    case ARC_RULE_BBOX_FILL:
        return 7;
    case ARC_RULE_TRANSLATE:
        return 8;
    case ARC_RULE_GRID_IDENTITY:
        return 9;
    case ARC_RULE_GRID_FOREGROUND_RECOLOR:
        return 10;
    }
    return 0;
}

static double arc_structural_program_score(const ARCTransformCandidate *candidate)
{
    switch (candidate->op) {
    case ARC_RULE_GRID_COLOR_MAP:
        return 120.0;
    case ARC_RULE_ROTATE90:
    case ARC_RULE_ROTATE180:
    case ARC_RULE_ROTATE270:
    case ARC_RULE_FLIP_HORIZONTAL:
    case ARC_RULE_FLIP_VERTICAL:
    case ARC_RULE_TRANSPOSE:
        return 110.0;
    case ARC_RULE_SCALE:
        return 105.0 - (double)candidate->scale;
    case ARC_RULE_INTERSECTION:
    case ARC_RULE_UNION:
    case ARC_RULE_OBJECT_SELECT:
        return 104.0;
    case ARC_RULE_COPY_LEFT:
    case ARC_RULE_COPY_RIGHT:
    case ARC_RULE_COLOR_MAP_LEFT:
        return 102.0;
    case ARC_RULE_CROP_BACKGROUND:
    case ARC_RULE_CROP_COLOR:
        return 100.0;
    case ARC_RULE_OBJECT_CROP_LARGEST:
    case ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR:
    case ARC_RULE_OBJECT_MASK_LARGEST:
        return 96.0;
    case ARC_RULE_CROP_BACKGROUND_RECOLOR:
    case ARC_RULE_CROP_COLOR_MASK:
        return 84.0;
    case ARC_RULE_BBOX_OUTLINE:
    case ARC_RULE_BBOX_FILL:
        return 78.0;
    case ARC_RULE_TRANSLATE:
        return 72.0 - (double)(abs(candidate->dx) + abs(candidate->dy));
    case ARC_RULE_GRID_IDENTITY:
        return 68.0;
    case ARC_RULE_GRID_FOREGROUND_RECOLOR:
        return 62.0;
    }
    return 0.0;
}

static int arc_candidate_to_program(const ARCTransformCandidate *candidate,
                                    NNProgram *program)
{
    if (candidate == NULL || program == NULL) {
        return 0;
    }

    if (arc_candidate_is_object_program(candidate)) {
        NNProgram components;
        NNProgram largest;
        NNProgram renderer;
        NNProgram selected;
        nn_program_init(&components,
                        "arc_agi",
                        "connected_components",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        100);
        nn_program_add_param(&components,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&largest,
                        "arc_agi",
                        "largest_component",
                        NN_PROGRAM_KIND_SELECTOR,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_OBJECT,
                        101);
        nn_program_init(&renderer,
                        "arc_agi",
                        candidate->op == ARC_RULE_OBJECT_MASK_LARGEST ? "object_mask" : "object_crop",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT,
                        NN_PROGRAM_VALUE_GRID,
                        candidate->op == ARC_RULE_OBJECT_MASK_LARGEST ? 102u : 103u);
        nn_program_add_param(&renderer,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        if (!nn_program_compose(&selected,
                                &components,
                                &largest,
                                "connected_components_then_largest") ||
            !nn_program_compose(program, &selected, &renderer, candidate->name)) {
            return 0;
        }
    } else {
        nn_program_init(program,
                        "arc_agi",
                        candidate->name,
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_GRID,
                        (uint32_t)candidate->op);
    }

    nn_program_add_param(program, "op", (double)candidate->op, (int64_t)candidate->op);
    nn_program_add_param(program,
                         "input_color",
                         (double)candidate->input_color,
                         candidate->input_color);
    nn_program_add_param(program,
                         "output_color",
                         (double)candidate->output_color,
                         candidate->output_color);
    nn_program_add_param(program,
                         "background",
                         (double)candidate->background_color,
                         candidate->background_color);
    nn_program_add_param(program, "dx", (double)candidate->dx, candidate->dx);
    nn_program_add_param(program, "dy", (double)candidate->dy, candidate->dy);
    nn_program_add_param(program, "scale", (double)candidate->scale, (int64_t)candidate->scale);
    if (candidate->op == ARC_RULE_GRID_COLOR_MAP) {
        for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
            char name[16];
            snprintf(name, sizeof(name), "map_%u", (unsigned)color);
            nn_program_add_param(program,
                                 name,
                                 (double)candidate->color_map[color],
                                 candidate->color_map[color]);
        }
    }
    nn_program_add_feature(program, (nn_real)((double)candidate->op / 32.0));
    nn_program_add_feature(program, (nn_real)((double)arc_candidate_family(candidate) / 16.0));
    nn_program_add_feature(program, (nn_real)((double)candidate->background_color / 9.0));
    nn_program_add_feature(program, (nn_real)((double)candidate->output_color / 9.0));
    nn_program_add_feature(program, (nn_real)(arc_candidate_is_object_program(candidate) ? 1.0 : 0.0));
    nn_program_add_feature(program, (nn_real)((double)(abs(candidate->dx) + abs(candidate->dy)) / 8.0));
    nn_program_add_feature(program, (nn_real)((double)candidate->scale / 4.0));
    nn_program_add_feature(program, (nn_real)(candidate->input_color == candidate->output_color ? 1.0 : 0.0));
    nn_program_refresh_canonical_id(program);
    return 1;
}

static int arc_encode_program_features(const ARCTransformCandidate *candidate,
                                       const ARCProgramTaskFeatures *task_features,
                                       nn_real *features,
                                       size_t capacity)
{
    if (candidate == NULL || task_features == NULL || features == NULL ||
        capacity < ARC_PROGRAM_FEATURE_COUNT) {
        return 0;
    }
    NNProgram program;
    size_t encoded_count = 0;
    if (!arc_candidate_to_program(candidate, &program) ||
        !nn_program_encode_features(&program,
                                    features,
                                    NN_PROGRAM_FEATURE_COUNT,
                                    &encoded_count) ||
        encoded_count != NN_PROGRAM_FEATURE_COUNT) {
        return 0;
    }
    arc_program_task_features_write(task_features,
                                    &features[NN_PROGRAM_FEATURE_COUNT]);
    return 1;
}

static int arc_build_program_feature_matrix(const ARCTransformCandidate *candidates,
                                            size_t candidate_count,
                                            const ARCProgramTaskFeatures *task_features,
                                            nn_real *features,
                                            size_t feature_count)
{
    if (feature_count != ARC_PROGRAM_FEATURE_COUNT) {
        return 0;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!arc_encode_program_features(&candidates[i],
                                         task_features,
                                         &features[i * feature_count],
                                         feature_count)) {
            return 0;
        }
    }
    return 1;
}

static size_t arc_evaluate_program_candidates(const ARCTask *task,
                                              const ARCTransformCandidate *candidates,
                                              size_t candidate_count,
                                              ARCProgramEval *evals)
{
    ARCGrid predictions[ARC_MAX_PAIRS];
    size_t task_cells = arc_task_test_cells(task);
    for (size_t i = 0; i < candidate_count; ++i) {
        NNProgram program;
        memset(&evals[i], 0, sizeof(evals[i]));
        if (arc_candidate_to_program(&candidates[i], &program)) {
            evals[i].program_hash = program.canonical_hash;
            evals[i].object_program = arc_candidate_is_object_program(&candidates[i]);
        }
        evals[i].train_consistent =
            arc_candidate_matches_train(task,
                                        &candidates[i],
                                        &evals[i].train_checked_cells,
                                        &evals[i].train_mismatches,
                                        &evals[i].unsupported_train_pairs);
        if (!evals[i].train_consistent) {
            continue;
        }
        evals[i].test_supported =
            arc_apply_candidate_to_tests(task,
                                         &candidates[i],
                                         predictions,
                                         &evals[i].unsupported_test_pairs);
        if (!evals[i].test_supported) {
            continue;
        }
        evals[i].prediction_hash = arc_prediction_hash(predictions, task->test_count);
        evals[i].test_cells = task_cells;
        for (size_t pair = 0; pair < task->test_count; ++pair) {
            if (!task->test[pair].has_output ||
                !arc_grid_same_shape(&predictions[pair], &task->test[pair].output)) {
                continue;
            }
            size_t cells = arc_grid_cells(&task->test[pair].output);
            for (size_t cell = 0; cell < cells; ++cell) {
                evals[i].test_correct_cells +=
                    predictions[pair].cells[cell] == task->test[pair].output.cells[cell] ? 1u : 0u;
            }
        }
    }
    return candidate_count;
}

static NNProgramProposerStrategy arc_program_strategy_to_nn(ARCProgramStrategy strategy)
{
    switch (strategy) {
    case ARC_PROGRAM_STRATEGY_GRAMMAR:
        return NN_PROGRAM_PROPOSER_GRAMMAR_ORDER;
    case ARC_PROGRAM_STRATEGY_STRUCTURAL:
        return NN_PROGRAM_PROPOSER_STRUCTURAL_HEURISTIC;
    case ARC_PROGRAM_STRATEGY_RANDOM:
        return NN_PROGRAM_PROPOSER_RANDOM;
    case ARC_PROGRAM_STRATEGY_LEARNED:
        return NN_PROGRAM_PROPOSER_LEARNED;
    case ARC_PROGRAM_STRATEGY_ORACLE:
        return NN_PROGRAM_PROPOSER_ORACLE;
    }
    return NN_PROGRAM_PROPOSER_GRAMMAR_ORDER;
}

static uint64_t arc_task_hash(const ARCTask *task)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *p = (const unsigned char *)task->id; *p != '\0'; ++p) {
        hash = arc_hash_mix(hash, *p);
    }
    return hash;
}

static NeuralNet *create_arc_program_model(unsigned int seed)
{
    static const size_t layers[] = {ARC_PROGRAM_FEATURE_COUNT, 64, 32, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int arc_program_learned_init(ARCProgramLearnedState *state, unsigned int seed)
{
    memset(state, 0, sizeof(*state));
    state->net = create_arc_program_model(seed);
    return state->net != NULL;
}

static void arc_program_learned_free(ARCProgramLearnedState *state)
{
    if (state == NULL) {
        return;
    }
    nn_free(state->net);
    memset(state, 0, sizeof(*state));
}

static double arc_learned_family_score(const ARCProgramLearnedState *state,
                                       const ARCTransformCandidate *candidate)
{
    size_t family = arc_candidate_family(candidate);
    if (state == NULL) {
        return 0.0;
    }
    if (family >= sizeof(state->family_score) / sizeof(state->family_score[0])) {
        return 0.0;
    }
    return state->family_score[family];
}

static double arc_learned_program_score(ARCProgramLearnedState *state,
                                        const ARCTransformCandidate *candidate,
                                        const ARCProgramTaskFeatures *task_features)
{
    if (state == NULL || state->net == NULL || state->trained_tasks == 0) {
        return arc_learned_family_score(state, candidate);
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    nn_real output[1];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return arc_learned_family_score(state, candidate);
    }
    nn_forward(state->net, features, output);
    return (double)output[0];
}

static int arc_program_eval_is_exact_success(const ARCProgramEval *eval)
{
    return eval->train_consistent &&
           eval->test_supported &&
           eval->test_cells > 0 &&
           eval->test_correct_cells == eval->test_cells;
}

static int arc_program_learned_train_task(ARCProgramLearnedState *state,
                                          const ARCTask *task,
                                          const ARCTransformCandidate *candidates,
                                          const ARCProgramEval *evals,
                                          size_t candidate_count,
                                          nn_real learning_rate)
{
    if (state == NULL || state->net == NULL || task == NULL || candidates == NULL ||
        evals == NULL || candidate_count == 0 || learning_rate <= (nn_real)0) {
        return 0;
    }

    size_t exact_count = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        exact_count += arc_program_eval_is_exact_success(&evals[i]) ? 1u : 0u;
    }
    if (exact_count == 0) {
        return 1;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    nn_real *features =
        calloc(candidate_count * ARC_PROGRAM_FEATURE_COUNT, sizeof(*features));
    if (features == NULL) {
        return 0;
    }
    if (!arc_build_program_feature_matrix(candidates,
                                          candidate_count,
                                          &task_features,
                                          features,
                                          ARC_PROGRAM_FEATURE_COUNT)) {
        free(features);
        return 0;
    }

    nn_real weight = (nn_real)(1.0 / (double)exact_count);
    for (size_t epoch = 0; epoch < ARC_PROGRAM_TRAIN_EPOCHS; ++epoch) {
        for (size_t i = 0; i < candidate_count; ++i) {
            if (!arc_program_eval_is_exact_success(&evals[i])) {
                continue;
            }
            nn_real loss = nn_train_softmax_scorer_weighted(state->net,
                                                            features,
                                                            candidate_count,
                                                            i,
                                                            weight,
                                                            learning_rate);
            if (!isfinite((double)loss)) {
                free(features);
                return 0;
            }
            ++state->train_updates;
        }
    }
    ++state->trained_tasks;
    free(features);
    return 1;
}

static void arc_build_program_ranks(const ARCTask *task,
                                    const ARCTransformCandidate *candidates,
                                    const ARCProgramEval *evals,
                                    size_t candidate_count,
                                    ARCProgramStrategy strategy,
                                    ARCProgramLearnedState *learned,
                                    const ARCProgramTaskFeatures *task_features,
                                    ARCProgramRank *ranks)
{
    uint64_t task_hash = arc_task_hash(task);
    for (size_t i = 0; i < candidate_count; ++i) {
        ranks[i].candidate_index = i;
        ranks[i].primary = 0.0;
        ranks[i].secondary = arc_structural_program_score(&candidates[i]);
        ranks[i].tie_breaker = arc_hash_mix(task_hash, evals[i].program_hash ^ (uint64_t)i);
        switch (strategy) {
        case ARC_PROGRAM_STRATEGY_GRAMMAR:
            ranks[i].primary = -(double)i;
            ranks[i].secondary = 0.0;
            ranks[i].tie_breaker = i;
            break;
        case ARC_PROGRAM_STRATEGY_STRUCTURAL:
            ranks[i].primary = ranks[i].secondary;
            break;
        case ARC_PROGRAM_STRATEGY_RANDOM:
            ranks[i].primary = (double)(ranks[i].tie_breaker & UINT64_C(0xffffffff));
            ranks[i].secondary = 0.0;
            break;
        case ARC_PROGRAM_STRATEGY_LEARNED:
            ranks[i].primary =
                ranks[i].secondary +
                arc_learned_program_score(learned, &candidates[i], task_features);
            break;
        case ARC_PROGRAM_STRATEGY_ORACLE:
            ranks[i].primary = evals[i].train_consistent ? (double)evals[i].test_correct_cells : -1.0;
            ranks[i].secondary = evals[i].test_cells != 0 &&
                                         evals[i].test_correct_cells == evals[i].test_cells
                                     ? 1.0
                                     : 0.0;
            break;
        }
    }

    for (size_t i = 1; i < candidate_count; ++i) {
        ARCProgramRank key = ranks[i];
        size_t j = i;
        while (j > 0) {
            ARCProgramRank prev = ranks[j - 1u];
            int before = key.primary > prev.primary ||
                         (key.primary == prev.primary && key.secondary > prev.secondary) ||
                         (key.primary == prev.primary && key.secondary == prev.secondary &&
                          key.tie_breaker < prev.tie_breaker);
            if (!before) {
                break;
            }
            ranks[j] = ranks[j - 1u];
            --j;
        }
        ranks[j] = key;
    }
}

static void arc_program_aggregate_init(ARCProgramProposerAggregate *aggregate,
                                       const char *task_name,
                                       ARCProgramStrategy strategy,
                                       size_t budget)
{
    memset(aggregate, 0, sizeof(*aggregate));
    aggregate->summary.domain = "arc_agi";
    aggregate->summary.task = task_name;
    aggregate->summary.strategy = arc_program_strategy_to_nn(strategy);
    aggregate->summary.budget = budget;
    aggregate->program_set_hash = UINT64_C(1469598103934665603);
}

static void arc_program_aggregate_task(ARCProgramProposerAggregate *aggregate,
                                       const ARCTask *task,
                                       const ARCTransformCandidate *candidates,
                                       const ARCProgramEval *evals,
                                       size_t candidate_count,
                                       ARCProgramStrategy strategy,
                                       ARCProgramLearnedState *learned,
                                       size_t budget)
{
    ARCProgramRank ranks[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramTaskFeatures task_features;
    size_t limit = candidate_count < budget ? candidate_count : budget;
    size_t task_cells = arc_task_test_cells(task);
    int have_prediction = 0;
    int unique_prediction = 1;
    uint64_t prediction_hash = 0;
    size_t prediction_correct_cells = 0;
    NNProgramCoverage coverage;
    uint64_t *covered = NULL;
    int have_coverage = nn_program_coverage_init(&coverage, candidate_count, task_cells);
    if (have_coverage && coverage.word_count != 0) {
        covered = calloc(coverage.word_count, sizeof(*covered));
        if (covered == NULL) {
            nn_program_coverage_free(&coverage);
            have_coverage = 0;
        }
    }
    if (have_coverage) {
        for (size_t i = 0; i < candidate_count; ++i) {
            for (size_t event = 0; event < evals[i].test_correct_cells; ++event) {
                nn_program_coverage_set(&coverage, i, event);
            }
        }
    }

    arc_program_task_features(task, &task_features);
    arc_build_program_ranks(task,
                            candidates,
                            evals,
                            candidate_count,
                            strategy,
                            learned,
                            &task_features,
                            ranks);
    aggregate->summary.event_count += task_cells;
    for (size_t rank = 0; rank < limit; ++rank) {
        size_t index = ranks[rank].candidate_index;
        ++aggregate->summary.proposed_programs;
        aggregate->program_set_hash =
            arc_hash_mix(aggregate->program_set_hash, evals[index].program_hash);
        if (!evals[index].train_consistent) {
            continue;
        }
        ++aggregate->summary.certified_programs;
        if (!evals[index].test_supported) {
            continue;
        }
        ++aggregate->summary.accepted_programs;
        if (!have_prediction) {
            prediction_hash = evals[index].prediction_hash;
            prediction_correct_cells =
                have_coverage && covered != NULL
                    ? nn_program_coverage_gain(&coverage, index, covered)
                    : evals[index].test_correct_cells;
            have_prediction = 1;
        } else if (prediction_hash != evals[index].prediction_hash) {
            unique_prediction = 0;
        }
    }

    if (have_prediction && unique_prediction) {
        ++aggregate->unique_prediction_tasks;
        aggregate->summary.covered_events += prediction_correct_cells;
        if (prediction_correct_cells == task_cells && task_cells > 0) {
            ++aggregate->exact_tasks;
        } else if (task_cells > prediction_correct_cells) {
            aggregate->summary.false_rejections += task_cells - prediction_correct_cells;
        }
    }
    if (aggregate->summary.event_count != 0) {
        aggregate->summary.utility =
            (double)aggregate->summary.covered_events /
            (double)aggregate->summary.event_count;
    }
    snprintf(aggregate->program_set_hash_text,
             sizeof(aggregate->program_set_hash_text),
             "%016llx",
             (unsigned long long)aggregate->program_set_hash);
    aggregate->summary.program_set_hash = aggregate->program_set_hash_text;
    free(covered);
    if (have_coverage) {
        nn_program_coverage_free(&coverage);
    }
}

static void arc_program_learned_observe(ARCProgramLearnedState *state,
                                        const ARCTask *task,
                                        const ARCTransformCandidate *candidates,
                                        const ARCProgramEval *evals,
                                        size_t candidate_count,
                                        nn_real learning_rate)
{
    size_t family_seen[32];
    size_t family_success[32];
    memset(family_seen, 0, sizeof(family_seen));
    memset(family_success, 0, sizeof(family_success));
    for (size_t i = 0; i < candidate_count; ++i) {
        size_t family = arc_candidate_family(&candidates[i]);
        if (family >= sizeof(family_seen) / sizeof(family_seen[0])) {
            continue;
        }
        ++family_seen[family];
        if (evals[i].train_consistent) {
            ++family_success[family];
        }
    }
    for (size_t family = 0; family < sizeof(family_seen) / sizeof(family_seen[0]); ++family) {
        if (family_seen[family] == 0) {
            continue;
        }
        double observation = (double)family_success[family] / (double)family_seen[family];
        state->family_score[family] = 0.85 * state->family_score[family] + 0.15 * observation;
    }
    arc_program_learned_train_task(state,
                                   task,
                                   candidates,
                                   evals,
                                   candidate_count,
                                   learning_rate);
}

static void arc_program_print_aggregate(const ARCProgramProposerAggregate *aggregate)
{
    nn_program_write_proposer_summary_json(stdout, &aggregate->summary);
}

static size_t arc_program_budget_list(const ARCConfig *config,
                                      size_t *budgets,
                                      size_t capacity)
{
    if (config != NULL && config->program_budget_grid) {
        size_t count = ARC_PROGRAM_BUDGET_GRID_COUNT < capacity
                           ? ARC_PROGRAM_BUDGET_GRID_COUNT
                           : capacity;
        for (size_t i = 0; i < count; ++i) {
            budgets[i] = ARC_PROGRAM_BUDGET_GRID[i];
        }
        return count;
    }
    if (capacity == 0) {
        return 0;
    }
    budgets[0] = config == NULL ? ARC_PROGRAM_PROPOSER_BUDGET : config->program_budget;
    return 1;
}

static void arc_emit_builtin_program_summaries(const ARCTask *task, const ARCConfig *config)
{
    ARCTransformCandidate candidates[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramEval evals[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramLearnedState learned;
    ARCProgramProposerAggregate aggregate;
    size_t budgets[ARC_PROGRAM_BUDGET_GRID_COUNT];
    size_t budget_count = arc_program_budget_list(config,
                                                  budgets,
                                                  ARC_PROGRAM_BUDGET_GRID_COUNT);
    if (!arc_program_learned_init(&learned,
                                  (unsigned int)((config == NULL ? 20240516u : config->seed) &
                                                 UINT32_MAX))) {
        return;
    }
    size_t candidate_count = arc_build_candidates(task, candidates, ARC_RULE_CANDIDATE_COUNT);
    arc_evaluate_program_candidates(task, candidates, candidate_count, evals);
    for (size_t budget_index = 0; budget_index < budget_count; ++budget_index) {
        for (size_t strategy = 0; strategy < ARC_PROGRAM_STRATEGY_COUNT; ++strategy) {
            arc_program_aggregate_init(&aggregate,
                                       task->id,
                                       (ARCProgramStrategy)strategy,
                                       budgets[budget_index]);
            arc_program_aggregate_task(&aggregate,
                                       task,
                                       candidates,
                                       evals,
                                       candidate_count,
                                       (ARCProgramStrategy)strategy,
                                       &learned,
                                       budgets[budget_index]);
            arc_program_print_aggregate(&aggregate);
        }
    }
    arc_program_learned_free(&learned);
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
    metrics.rules = arc_build_candidates(task, candidates, ARC_RULE_CANDIDATE_COUNT);

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

static int arc_pretrain_program_proposer(ARCProgramLearnedState *learned_state,
                                         const char *task_dir,
                                         size_t max_tasks,
                                         nn_real learning_rate)
{
    ARCTaskFile *files = calloc(ARC_MAX_TASK_FILES, sizeof(*files));
    if (files == NULL) {
        return 0;
    }
    size_t file_count = 0;
    if (!arc_collect_task_files(task_dir, files, ARC_MAX_TASK_FILES, &file_count)) {
        fprintf(stderr, "failed to read ARC program training directory: %s\n", task_dir);
        free(files);
        return 0;
    }

    size_t loaded = 0;
    for (size_t i = 0; i < file_count; ++i) {
        if (max_tasks != 0 && loaded >= max_tasks) {
            break;
        }
        ARCTask task;
        if (!arc_load_task_file(files[i].path, files[i].id, &task)) {
            continue;
        }
        ++loaded;
        ARCTransformCandidate candidates[ARC_RULE_CANDIDATE_COUNT];
        ARCProgramEval evals[ARC_RULE_CANDIDATE_COUNT];
        size_t candidate_count = arc_build_candidates(&task,
                                                      candidates,
                                                      ARC_RULE_CANDIDATE_COUNT);
        arc_evaluate_program_candidates(&task, candidates, candidate_count, evals);
        if (!arc_program_learned_train_task(learned_state,
                                            &task,
                                            candidates,
                                            evals,
                                            candidate_count,
                                            learning_rate)) {
            free(files);
            return 0;
        }
    }
    free(files);
    return loaded > 0;
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
    if (transform->rules > summary->rules) {
        summary->rules = transform->rules;
    }
    summary->train_consistent_rules += transform->train_consistent_rules;
    summary->unique_prediction_tasks += transform->unique_test_prediction ? 1u : 0u;
    summary->scored_tasks += transform->test_cells > 0 ? 1u : 0u;
    summary->test_exact_tasks += transform->test_exact ? 1u : 0u;
    summary->certified_tasks += transform->certified ? 1u : 0u;
    summary->unsupported_tasks += transform->train_checked_cells == 0 ? 1u : 0u;
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
    size_t budgets[ARC_PROGRAM_BUDGET_GRID_COUNT];
    size_t budget_count = arc_program_budget_list(config,
                                                  budgets,
                                                  ARC_PROGRAM_BUDGET_GRID_COUNT);
    ARCProgramProposerAggregate
        program_aggregates[ARC_PROGRAM_BUDGET_GRID_COUNT][ARC_PROGRAM_STRATEGY_COUNT];
    ARCProgramLearnedState learned_state;
    if (!arc_program_learned_init(&learned_state,
                                  (unsigned int)(config->seed & UINT32_MAX))) {
        free(files);
        return 0;
    }
    if (config->program_train_dir != NULL &&
        !arc_pretrain_program_proposer(&learned_state,
                                       config->program_train_dir,
                                       config->program_train_max_tasks,
                                       config->program_learning_rate)) {
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
    }
    for (size_t budget_index = 0; budget_index < budget_count; ++budget_index) {
        for (size_t strategy = 0; strategy < ARC_PROGRAM_STRATEGY_COUNT; ++strategy) {
            arc_program_aggregate_init(&program_aggregates[budget_index][strategy],
                                       "arc_loader",
                                       (ARCProgramStrategy)strategy,
                                       budgets[budget_index]);
        }
    }

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
        ARCTransformCandidate candidates[ARC_RULE_CANDIDATE_COUNT];
        ARCProgramEval evals[ARC_RULE_CANDIDATE_COUNT];
        size_t candidate_count = arc_build_candidates(&task,
                                                      candidates,
                                                      ARC_RULE_CANDIDATE_COUNT);
        arc_evaluate_program_candidates(&task, candidates, candidate_count, evals);
        for (size_t budget_index = 0; budget_index < budget_count; ++budget_index) {
            for (size_t strategy = 0; strategy < ARC_PROGRAM_STRATEGY_COUNT; ++strategy) {
                arc_program_aggregate_task(&program_aggregates[budget_index][strategy],
                                           &task,
                                           candidates,
                                           evals,
                                           candidate_count,
                                           (ARCProgramStrategy)strategy,
                                           &learned_state,
                                           budgets[budget_index]);
            }
        }
        if (config->program_train_dir == NULL) {
            arc_program_learned_observe(&learned_state,
                                        &task,
                                        candidates,
                                        evals,
                                        candidate_count,
                                        config->program_learning_rate);
        }
        if (!config->loader_summary_only) {
            print_arc_transform_json(task.id, &transform);
            if (transform.certified) {
                ARCMetrics transform_metrics = arc_transform_to_metrics(&task, &transform);
                print_arc_metrics_json(task.id,
                                       "symbolic-transform-v4",
                                       "certified_transform_gate",
                                       "arc-transform-gate-v4",
                                       &transform_metrics);
            }
        }
    }
    print_arc_loader_summary_json(config->task_dir, &summary);
    for (size_t budget_index = 0; budget_index < budget_count; ++budget_index) {
        for (size_t strategy = 0; strategy < ARC_PROGRAM_STRATEGY_COUNT; ++strategy) {
            arc_program_print_aggregate(&program_aggregates[budget_index][strategy]);
        }
    }
    arc_program_learned_free(&learned_state);
    free(files);
    return summary.loaded_tasks > 0;
}

static void print_usage(FILE *stream)
{
    fputs("usage: arc_agi [--schema] [--epochs count] [--steps-per-epoch count] "
          "[--test-samples count] [--learning-rate rate] [--seed seed] [--save path] "
          "[--task-dir path] [--max-tasks count] [--program-budget count] "
          "[--program-budget-grid] [--program-train-dir path] "
          "[--program-train-max-tasks count] [--program-learning-rate rate] "
          "[--loader-summary-only]\n",
          stream);
}

static int parse_config(int argc, char **argv, ARCConfig *config, int *schema_only)
{
    config->epochs = ARC_DEFAULT_EPOCHS;
    config->steps_per_epoch = ARC_DEFAULT_STEPS_PER_EPOCH;
    config->test_samples = ARC_DEFAULT_TEST_SAMPLES;
    config->max_tasks = 0;
    config->program_budget = ARC_PROGRAM_PROPOSER_BUDGET;
    config->program_train_max_tasks = 0;
    config->seed = 20240516u;
    config->learning_rate = (nn_real)0.006;
    config->program_learning_rate = (nn_real)0.015;
    config->save_path = NULL;
    config->task_dir = NULL;
    config->program_train_dir = NULL;
    config->loader_summary_only = 0;
    config->program_budget_grid = 0;
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
        } else if (strcmp(argv[i], "--program-budget") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->program_budget)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--program-budget-grid") == 0) {
            config->program_budget_grid = 1;
        } else if (strcmp(argv[i], "--program-train-dir") == 0 && i + 1 < argc) {
            config->program_train_dir = argv[++i];
            if (config->program_train_dir[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--program-train-max-tasks") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &config->program_train_max_tasks)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--program-learning-rate") == 0 && i + 1 < argc) {
            if (!parse_real_text(argv[++i], &config->program_learning_rate)) {
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
                           "symbolic-transform-v4",
                           "certified_transform_gate",
                           "arc-transform-gate-v4",
                           &transform_metrics);
    print_arc_metrics_line("symbolic-transform-v4",
                           "certified_transform_gate",
                           &transform_metrics);
    arc_emit_builtin_program_summaries(&builtin_task, &config);
    nn_free(net);
    return EXIT_SUCCESS;
}
