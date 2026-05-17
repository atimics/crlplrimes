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
    ARC_MAX_SCENE_OBJECTS = 64,
    ARC_MAX_SCENE_RELATION_PAIRS = ARC_MAX_SCENE_OBJECTS * (ARC_MAX_SCENE_OBJECTS - 1) / 2,
    ARC_MAX_PAIRS = 16,
    ARC_MAX_TASK_ID = 64,
    ARC_MAX_TASK_FILES = 4096,
    ARC_MAX_TASK_PATH = 1024,
    ARC_ACCEPTED_RULES_BYTES = 2048,
    ARC_TRANSFORM_NAME_BYTES = 128,
    ARC_RULE_CANDIDATE_COUNT = 4096,
    ARC_GRAMMAR_VERSION = 10,
    ARC_LEARNED_COMPOSE_TOP = 16,
    ARC_LEARNED_COMPOSE_LIMIT = 256,
    ARC_HYPOTHESIS_PROPOSAL_CAPACITY = 256,
    ARC_HYPOTHESIS_TOP_K = 32,
    ARC_HYPOTHESIS_STRUCTURAL_SCORE = 130,
    ARC_HYPOTHESIS_TEMPLATE_BUCKETS = 8,
    ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS = 9,
    ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS = 32,
    ARC_HYPOTHESIS_PRIOR_MIN_PROPOSALS = 3,
    ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT = NN_HYPOTHESIS_FEATURE_COUNT,
    ARC_HYPOTHESIS_PRIOR_EPOCHS = 32,
    ARC_PROGRAM_PROPOSER_BUDGET = 64,
    ARC_PROGRAM_BUDGET_GRID_COUNT = 4,
    ARC_PROGRAM_TASK_FEATURE_COUNT = 32,
    ARC_PROGRAM_FEATURE_COUNT = NN_PROGRAM_FEATURE_COUNT + ARC_PROGRAM_TASK_FEATURE_COUNT,
    ARC_PROGRAM_TRAIN_EPOCHS = 5,
    ARC_PROGRAM_STRATEGY_COUNT = 5,
    ARC_OBJECT_CONNECT_HORIZONTAL = 0,
    ARC_OBJECT_CONNECT_VERTICAL = 1,
    ARC_OBJECT_CONNECT_BOTH = 2,
    ARC_OBJECT_RELATION_TRANSLATE = 1,
    ARC_OBJECT_RELATION_CONNECT_ALIGNED = 2,
    ARC_OBJECT_RELATION_FILL_ENCLOSED = 3,
    ARC_OBJECT_RELATION_KIND_COUNT = 4,
    ARC_OBJECT_SELECTOR_ALL_COLOR = 0,
    ARC_OBJECT_SELECTOR_UNIQUE_COLOR = 1,
    ARC_OBJECT_SELECTOR_LARGEST_COLOR = 2,
    ARC_OBJECT_SELECTOR_SMALLEST_COLOR = 3,
    ARC_OBJECT_SELECTOR_INTERIOR_COLOR = 4,
    ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR = 5,
    ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR = 6,
    ARC_OBJECT_SELECTOR_KIND_COUNT = 7,
    ARC_OBJECT_RENDER_OVERLAY = 0,
    ARC_OBJECT_RENDER_ONLY = 1,
    ARC_OBJECT_RENDER_KIND_COUNT = 2,
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
    ARC_RULE_COMPOSED = 26,
    ARC_RULE_OBJECT_TRANSLATE = 27,
    ARC_RULE_OBJECT_CONNECT_ALIGNED = 28,
    ARC_RULE_OBJECT_FILL_ENCLOSED = 29,
    ARC_RULE_OBJECT_RELATION_INDUCED = 30,
    ARC_RULE_HYPOTHESIS_APPLIED = 31,
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
        "connected_component_largest, object_relation_induced, learned_scale_composition",
    .certificate_checker =
        "train-pair consistency plus unique grammar-consistent test prediction, reported against compiled test output",
    .gate_executor = "certified transform gate emits exact color candidates when the grammar prediction is unique",
    .proposer_inputs =
        "typed NNProgram transform descriptors encoded by nn_program_encode_features, "
        "plus ARC train-pair shape/color/object-scene selector and parameter descriptors",
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
    size_t min_induction_evidence;
    uint64_t seed;
    nn_real learning_rate;
    nn_real program_learning_rate;
    const char *save_path;
    const char *task_dir;
    const char *program_train_dir;
    int loader_summary_only;
    int program_budget_grid;
    int induction_smoke;
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
    size_t id;
    uint8_t color;
    size_t count;
    size_t min_x;
    size_t min_y;
    size_t max_x;
    size_t max_y;
    double centroid_x;
    double centroid_y;
    uint64_t shape_hash;
    uint8_t touches_border;
} ARCObject;

typedef struct {
    uint8_t background_color;
    size_t object_count;
    size_t repeated_shape_pairs;
    size_t aligned_pairs;
    size_t adjacent_pairs;
    ARCObject objects[ARC_MAX_SCENE_OBJECTS];
} ARCScene;

typedef struct {
    ARCTransformRule op;
    uint8_t input_color;
    uint8_t output_color;
    uint8_t background_color;
    uint8_t selector_kind;
    uint8_t output_selector_kind;
    uint8_t reference_color;
    uint8_t render_mode;
    uint8_t color_map[ARC_COLOR_COUNT];
    int dx;
    int dy;
    size_t scale;
    char name[ARC_TRANSFORM_NAME_BYTES];
} ARCTransformStep;

typedef struct {
    ARCTransformRule op;
    uint8_t input_color;
    uint8_t output_color;
    uint8_t background_color;
    uint8_t selector_kind;
    uint8_t output_selector_kind;
    uint8_t reference_color;
    uint8_t render_mode;
    uint8_t color_map[ARC_COLOR_COUNT];
    int dx;
    int dy;
    size_t scale;
    size_t step_count;
    ARCTransformStep steps[2];
    uint32_t hypothesis_template_id;
    uint64_t hypothesis_canonical_hash;
    uint8_t hypothesis_slot_count;
    int64_t hypothesis_slot_integers[NN_HYPOTHESIS_MAX_SLOTS];
    char name[ARC_TRANSFORM_NAME_BYTES];
} ARCTransformCandidate;

typedef struct {
    NNHypothesis hypothesis;
    ARCTransformCandidate candidate;
    double score;
} ARCHypothesisProposal;

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
    NeuralNet *hypothesis_prior_net;
    size_t trained_tasks;
    size_t train_updates;
    size_t hypothesis_prior_samples;
    size_t hypothesis_prior_updates;
    double hypothesis_prior_loss;
    double family_score[32];
    double relation_score[ARC_OBJECT_RELATION_KIND_COUNT];
    double relation_color_score[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    double relation_background_score[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT];
    double relation_selector_score[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    double relation_output_selector_score[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    double relation_render_score[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_RENDER_KIND_COUNT];
    double object_selector_score[ARC_OBJECT_SELECTOR_KIND_COUNT];
    double object_color_score[ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    double object_render_score[ARC_OBJECT_RENDER_KIND_COUNT];
    size_t hypothesis_slot_proposal_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS]
                                         [ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS]
                                         [ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS];
    size_t hypothesis_slot_certified_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS]
                                          [ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS]
                                          [ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS];
} ARCProgramLearnedState;

typedef struct ARCHypothesisTemplateBinding ARCHypothesisTemplateBinding;

typedef const NNHypothesisTemplate *(*ARCHypothesisTemplateFn)(void);
typedef size_t (*ARCHypothesisInduceFn)(const ARCHypothesisTemplateBinding *binding,
                                        const ARCTask *task,
                                        ARCProgramLearnedState *learned,
                                        const ARCTransformCandidate *existing_candidates,
                                        size_t existing_count,
                                        ARCHypothesisProposal *proposals,
                                        size_t proposal_capacity,
                                        size_t proposal_count);
typedef int (*ARCHypothesisToCandidateFn)(const NNHypothesis *hypothesis,
                                          ARCTransformCandidate *candidate);
typedef int (*ARCHypothesisApplyFn)(const NNHypothesis *hypothesis,
                                    const ARCGrid *input,
                                    ARCGrid *output);

struct ARCHypothesisTemplateBinding {
    const char *name;
    uint32_t template_id;
    ARCHypothesisTemplateFn template_fn;
    ARCHypothesisInduceFn induce;
    ARCHypothesisToCandidateFn to_candidate;
    ARCHypothesisApplyFn apply;
};

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
    double input_object_count;
    double output_object_count;
    double input_repeated_shape_pairs;
    double output_repeated_shape_pairs;
    double input_aligned_pairs;
    double output_aligned_pairs;
    double input_adjacent_pairs;
    double output_adjacent_pairs;
} ARCProgramTaskFeatures;

_Static_assert(ARC_PROGRAM_TASK_FEATURE_COUNT == 32,
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

static int arc_extract_scene(const ARCGrid *grid, uint8_t background, ARCScene *scene);
static void print_json_string(const char *text);
static int arc_scene_unique_color_object(const ARCScene *scene,
                                         uint8_t color,
                                         ARCObject *object_out);
static int arc_apply_candidate(const ARCTransformCandidate *candidate,
                               const ARCGrid *input,
                               ARCGrid *output);
static int arc_apply_hypothesis_candidate(const ARCTransformCandidate *candidate,
                                          const ARCGrid *input,
                                          ARCGrid *output);
static const NNHypothesisTemplate *arc_hypothesis_template_object_translate(void);
static const NNHypothesisTemplate *arc_hypothesis_template_move_to_anchor(void);
static const ARCHypothesisTemplateBinding *arc_hypothesis_template_registry(size_t *count);
static const ARCHypothesisTemplateBinding *arc_hypothesis_binding_for_template_id(uint32_t template_id);
static int arc_hypothesis_from_candidate(const ARCTransformCandidate *candidate,
                                         NNHypothesis *hypothesis);
static int arc_candidate_from_translate_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCTransformCandidate *candidate);
static int arc_candidate_from_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                                       ARCTransformCandidate *candidate);
static int arc_apply_translate_hypothesis(const NNHypothesis *hypothesis,
                                          const ARCGrid *input,
                                          ARCGrid *output);
static int arc_apply_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                               const ARCGrid *input,
                                               ARCGrid *output);
static size_t arc_induce_translate_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                              const ARCTask *task,
                                              ARCProgramLearnedState *learned,
                                              const ARCTransformCandidate *existing_candidates,
                                              size_t existing_count,
                                              ARCHypothesisProposal *proposals,
                                              size_t proposal_capacity,
                                              size_t proposal_count);
static size_t arc_induce_move_to_anchor_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                   const ARCTask *task,
                                                   ARCProgramLearnedState *learned,
                                                   const ARCTransformCandidate *existing_candidates,
                                                   size_t existing_count,
                                                   ARCHypothesisProposal *proposals,
                                                   size_t proposal_capacity,
                                                   size_t proposal_count);

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

static int arc_store_hypothesis_candidate(ARCTransformCandidate *candidate,
                                          const NNHypothesis *hypothesis)
{
    if (candidate == NULL || hypothesis == NULL ||
        hypothesis->template_ref == NULL ||
        !nn_hypothesis_is_fully_bound(hypothesis) ||
        hypothesis->slot_count > NN_HYPOTHESIS_MAX_SLOTS) {
        return 0;
    }
    candidate->op = ARC_RULE_HYPOTHESIS_APPLIED;
    candidate->hypothesis_template_id = hypothesis->template_ref->template_id;
    candidate->hypothesis_canonical_hash = hypothesis->canonical_hash;
    candidate->hypothesis_slot_count = (uint8_t)hypothesis->slot_count;
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        candidate->hypothesis_slot_integers[i] = hypothesis->slots[i].integer;
    }
    return 1;
}

static int arc_hypothesis_from_candidate(const ARCTransformCandidate *candidate,
                                         NNHypothesis *hypothesis)
{
    if (candidate == NULL || hypothesis == NULL ||
        candidate->op != ARC_RULE_HYPOTHESIS_APPLIED) {
        return 0;
    }
    const ARCHypothesisTemplateBinding *binding =
        arc_hypothesis_binding_for_template_id(candidate->hypothesis_template_id);
    if (binding == NULL || binding->template_fn == NULL ||
        !nn_hypothesis_init(hypothesis, binding->template_fn()) ||
        candidate->hypothesis_slot_count != hypothesis->slot_count) {
        return 0;
    }
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotSpec *slot = &hypothesis->template_ref->slots[i];
        int64_t integer = candidate->hypothesis_slot_integers[i];
        if (!nn_hypothesis_bind(hypothesis,
                                slot->name,
                                (double)integer,
                                integer)) {
            return 0;
        }
    }
    return hypothesis->canonical_hash == candidate->hypothesis_canonical_hash;
}

static int arc_candidate_is_composed(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           candidate->op == ARC_RULE_COMPOSED &&
           candidate->step_count == 2;
}

static void arc_step_from_candidate(const ARCTransformCandidate *candidate, ARCTransformStep *step)
{
    memset(step, 0, sizeof(*step));
    step->op = candidate->op;
    step->input_color = candidate->input_color;
    step->output_color = candidate->output_color;
    step->background_color = candidate->background_color;
    step->selector_kind = candidate->selector_kind;
    step->output_selector_kind = candidate->output_selector_kind;
    step->reference_color = candidate->reference_color;
    step->render_mode = candidate->render_mode;
    memcpy(step->color_map, candidate->color_map, sizeof(step->color_map));
    step->dx = candidate->dx;
    step->dy = candidate->dy;
    step->scale = candidate->scale;
    snprintf(step->name, sizeof(step->name), "%s", candidate->name);
}

static void arc_candidate_from_step(const ARCTransformStep *step, ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, step->op, step->name);
    candidate->input_color = step->input_color;
    candidate->output_color = step->output_color;
    candidate->background_color = step->background_color;
    candidate->selector_kind = step->selector_kind;
    candidate->output_selector_kind = step->output_selector_kind;
    candidate->reference_color = step->reference_color;
    candidate->render_mode = step->render_mode;
    memcpy(candidate->color_map, step->color_map, sizeof(candidate->color_map));
    candidate->dx = step->dx;
    candidate->dy = step->dy;
    candidate->scale = step->scale;
}

static int arc_candidate_is_object_relation(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           (candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED ||
            candidate->op == ARC_RULE_OBJECT_TRANSLATE ||
            candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED ||
            candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED);
}

static size_t arc_object_relation_kind(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL) {
        return 0;
    }
    if (candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED) {
        return candidate->scale;
    }
    if (candidate->op == ARC_RULE_OBJECT_TRANSLATE) {
        return ARC_OBJECT_RELATION_TRANSLATE;
    }
    if (candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED) {
        return ARC_OBJECT_RELATION_CONNECT_ALIGNED;
    }
    if (candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED) {
        return ARC_OBJECT_RELATION_FILL_ENCLOSED;
    }
    return 0;
}

static const char *arc_object_relation_kind_name(size_t kind)
{
    switch (kind) {
    case ARC_OBJECT_RELATION_TRANSLATE:
        return "translate";
    case ARC_OBJECT_RELATION_CONNECT_ALIGNED:
        return "connect_aligned";
    case ARC_OBJECT_RELATION_FILL_ENCLOSED:
        return "fill_enclosed";
    }
    return "unknown";
}

static const char *arc_object_selector_short_name(uint8_t selector_kind)
{
    switch (selector_kind) {
    case ARC_OBJECT_SELECTOR_UNIQUE_COLOR:
        return "uniq";
    case ARC_OBJECT_SELECTOR_LARGEST_COLOR:
        return "max";
    case ARC_OBJECT_SELECTOR_SMALLEST_COLOR:
        return "min";
    case ARC_OBJECT_SELECTOR_INTERIOR_COLOR:
        return "inner";
    case ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR:
        return "edge";
    case ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR:
        return "near";
    case ARC_OBJECT_SELECTOR_ALL_COLOR:
    default:
        return "all";
    }
}

static uint8_t arc_candidate_selector_kind(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL) {
        return ARC_OBJECT_SELECTOR_ALL_COLOR;
    }
    if (candidate->selector_kind != 0) {
        return candidate->selector_kind;
    }
    if (candidate->op == ARC_RULE_OBJECT_TRANSLATE ||
        arc_object_relation_kind(candidate) == ARC_OBJECT_RELATION_TRANSLATE ||
        candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED ||
        arc_object_relation_kind(candidate) == ARC_OBJECT_RELATION_FILL_ENCLOSED) {
        return ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    }
    return ARC_OBJECT_SELECTOR_ALL_COLOR;
}

static uint8_t arc_candidate_output_selector_kind(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL || candidate->output_selector_kind == 0) {
        return ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    }
    return candidate->output_selector_kind;
}

static uint8_t arc_candidate_render_mode(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL ||
        candidate->render_mode >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return ARC_OBJECT_RENDER_OVERLAY;
    }
    return candidate->render_mode;
}

static int arc_rule_can_end_composition(ARCTransformRule op)
{
    switch (op) {
    case ARC_RULE_GRID_IDENTITY:
    case ARC_RULE_GRID_COLOR_MAP:
    case ARC_RULE_GRID_FOREGROUND_RECOLOR:
    case ARC_RULE_CROP_BACKGROUND:
    case ARC_RULE_CROP_BACKGROUND_RECOLOR:
    case ARC_RULE_CROP_COLOR:
    case ARC_RULE_CROP_COLOR_MASK:
    case ARC_RULE_ROTATE90:
    case ARC_RULE_ROTATE180:
    case ARC_RULE_ROTATE270:
    case ARC_RULE_FLIP_HORIZONTAL:
    case ARC_RULE_FLIP_VERTICAL:
    case ARC_RULE_TRANSPOSE:
    case ARC_RULE_SCALE:
    case ARC_RULE_BBOX_OUTLINE:
    case ARC_RULE_BBOX_FILL:
    case ARC_RULE_TRANSLATE:
    case ARC_RULE_OBJECT_CROP_LARGEST:
    case ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR:
    case ARC_RULE_OBJECT_MASK_LARGEST:
    case ARC_RULE_OBJECT_TRANSLATE:
    case ARC_RULE_OBJECT_CONNECT_ALIGNED:
    case ARC_RULE_OBJECT_FILL_ENCLOSED:
    case ARC_RULE_OBJECT_RELATION_INDUCED:
        return 1;
    case ARC_RULE_HYPOTHESIS_APPLIED:
    case ARC_RULE_COPY_LEFT:
    case ARC_RULE_COPY_RIGHT:
    case ARC_RULE_INTERSECTION:
    case ARC_RULE_UNION:
    case ARC_RULE_OBJECT_SELECT:
    case ARC_RULE_COLOR_MAP_LEFT:
    case ARC_RULE_COMPOSED:
        return 0;
    }
    return 0;
}

static int arc_init_composed_candidate(ARCTransformCandidate *candidate,
                                       const ARCTransformCandidate *first,
                                       const ARCTransformCandidate *second)
{
    if (candidate == NULL || first == NULL || second == NULL ||
        arc_candidate_is_composed(first) || arc_candidate_is_composed(second) ||
        !arc_rule_can_end_composition(second->op) ||
        (first->op != ARC_RULE_SCALE && second->op != ARC_RULE_SCALE)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_COMPOSED, "");
    candidate->step_count = 2;
    candidate->input_color = first->input_color;
    candidate->output_color = second->output_color;
    candidate->background_color = first->background_color;
    candidate->selector_kind = first->selector_kind;
    candidate->output_selector_kind = second->output_selector_kind;
    candidate->reference_color = first->reference_color;
    candidate->render_mode = second->render_mode;
    candidate->dx = first->dx + second->dx;
    candidate->dy = first->dy + second->dy;
    candidate->scale = second->scale;
    arc_step_from_candidate(first, &candidate->steps[0]);
    arc_step_from_candidate(second, &candidate->steps[1]);
    snprintf(candidate->name,
             sizeof(candidate->name),
             "compose_%s_then_%s",
             first->name,
             second->name);
    return 1;
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

static uint8_t arc_grid_dominant_color(const ARCGrid *grid)
{
    size_t counts[ARC_COLOR_COUNT];
    memset(counts, 0, sizeof(counts));
    for (size_t cell = 0; cell < arc_grid_cells(grid); ++cell) {
        if (grid->cells[cell] < ARC_COLOR_COUNT) {
            ++counts[grid->cells[cell]];
        }
    }
    uint8_t best_color = 0;
    size_t best_count = counts[0];
    for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
        if (counts[color] > best_count) {
            best_count = counts[color];
            best_color = color;
        }
    }
    return best_color;
}

static void arc_add_unique_color(uint8_t *colors, size_t capacity, size_t *count, uint8_t color)
{
    if (colors == NULL || count == NULL || *count >= capacity) {
        return;
    }
    for (size_t i = 0; i < *count; ++i) {
        if (colors[i] == color) {
            return;
        }
    }
    colors[*count] = color;
    ++*count;
}

static size_t arc_task_relation_backgrounds(const ARCTask *task,
                                            uint8_t *backgrounds,
                                            size_t capacity)
{
    size_t count = 0;
    arc_add_unique_color(backgrounds, capacity, &count, 0);
    if (task == NULL || task->train_count == 0) {
        return count;
    }

    int have_background = 0;
    uint8_t observed_background = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output) {
            return count;
        }
        uint8_t input_background = arc_grid_dominant_color(&task->train[pair].input);
        uint8_t output_background = arc_grid_dominant_color(&task->train[pair].output);
        if (input_background != output_background) {
            return count;
        }
        if (!have_background) {
            observed_background = input_background;
            have_background = 1;
        } else if (observed_background != input_background) {
            return count;
        }
    }
    if (have_background) {
        arc_add_unique_color(backgrounds, capacity, &count, observed_background);
    }
    return count;
}

static int arc_objects_same_shape(const ARCObject *lhs, const ARCObject *rhs)
{
    size_t lhs_width = lhs->max_x - lhs->min_x + 1u;
    size_t lhs_height = lhs->max_y - lhs->min_y + 1u;
    size_t rhs_width = rhs->max_x - rhs->min_x + 1u;
    size_t rhs_height = rhs->max_y - rhs->min_y + 1u;
    return lhs->count == rhs->count &&
           lhs_width == rhs_width &&
           lhs_height == rhs_height &&
           lhs->shape_hash == rhs->shape_hash;
}

static size_t arc_scene_count_color_objects(const ARCScene *scene, uint8_t color)
{
    size_t count = 0;
    for (size_t i = 0; i < scene->object_count; ++i) {
        count += scene->objects[i].color == color ? 1u : 0u;
    }
    return count;
}

static double arc_object_distance2(const ARCObject *lhs, const ARCObject *rhs)
{
    double dx = lhs->centroid_x - rhs->centroid_x;
    double dy = lhs->centroid_y - rhs->centroid_y;
    return dx * dx + dy * dy;
}

static int arc_select_scene_object(const ARCScene *scene,
                                   uint8_t selector_kind,
                                   uint8_t color,
                                   uint8_t reference_color,
                                   ARCObject *object_out)
{
    if (selector_kind == ARC_OBJECT_SELECTOR_ALL_COLOR) {
        selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    }
    if (selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR) {
        return arc_scene_unique_color_object(scene, color, object_out);
    }

    ARCObject reference;
    if (selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR &&
        !arc_scene_unique_color_object(scene, reference_color, &reference)) {
        return 0;
    }

    int found = 0;
    int tied = 0;
    ARCObject best;
    double best_score = 0.0;
    memset(&best, 0, sizeof(best));
    for (size_t i = 0; i < scene->object_count; ++i) {
        const ARCObject *object = &scene->objects[i];
        double score = 0.0;
        if (object->color != color) {
            continue;
        }
        switch (selector_kind) {
        case ARC_OBJECT_SELECTOR_LARGEST_COLOR:
            score = (double)object->count;
            break;
        case ARC_OBJECT_SELECTOR_SMALLEST_COLOR:
            score = -(double)object->count;
            break;
        case ARC_OBJECT_SELECTOR_INTERIOR_COLOR:
            if (object->touches_border) {
                continue;
            }
            score = 0.0;
            break;
        case ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR:
            if (!object->touches_border) {
                continue;
            }
            score = 0.0;
            break;
        case ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR:
            if (object->id == reference.id) {
                continue;
            }
            score = -arc_object_distance2(object, &reference);
            break;
        default:
            return 0;
        }
        if (!found || score > best_score) {
            best = *object;
            best_score = score;
            found = 1;
            tied = 0;
        } else if (fabs(score - best_score) < 1e-9) {
            tied = 1;
        }
    }
    if (!found || tied) {
        return 0;
    }
    if (object_out != NULL) {
        *object_out = best;
    }
    return 1;
}

static int arc_infer_selector_for_scene_object(const ARCScene *scene,
                                               const ARCObject *target,
                                               uint8_t *selector_kind_out,
                                               uint8_t *reference_color_out)
{
    if (scene == NULL || target == NULL ||
        selector_kind_out == NULL || reference_color_out == NULL) {
        return 0;
    }
    static const uint8_t selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t i = 0; i < sizeof(selectors) / sizeof(selectors[0]); ++i) {
        ARCObject selected;
        if (arc_select_scene_object(scene,
                                    selectors[i],
                                    target->color,
                                    0,
                                    &selected) &&
            selected.id == target->id) {
            *selector_kind_out = selectors[i];
            *reference_color_out = 0;
            return 1;
        }
    }
    for (uint8_t reference_color = 0;
         reference_color < ARC_COLOR_COUNT;
         ++reference_color) {
        ARCObject selected;
        if (reference_color == scene->background_color ||
            reference_color == target->color ||
            !arc_scene_unique_color_object(scene, reference_color, NULL)) {
            continue;
        }
        if (arc_select_scene_object(scene,
                                    ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                    target->color,
                                    reference_color,
                                    &selected) &&
            selected.id == target->id) {
            *selector_kind_out = ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR;
            *reference_color_out = reference_color;
            return 1;
        }
    }
    return 0;
}

static int arc_infer_non_reference_selector_for_scene_object(const ARCScene *scene,
                                                            const ARCObject *target,
                                                            uint8_t *selector_kind_out)
{
    uint8_t selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    uint8_t reference_color = 0;
    if (selector_kind_out == NULL ||
        !arc_infer_selector_for_scene_object(scene,
                                             target,
                                             &selector_kind,
                                             &reference_color) ||
        selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) {
        return 0;
    }
    *selector_kind_out = selector_kind;
    return 1;
}

static int arc_build_object_translation_candidate(const ARCTask *task,
                                                  uint8_t background,
                                                  uint8_t input_color,
                                                  uint8_t output_color,
                                                  uint8_t selector_kind,
                                                  uint8_t output_selector_kind,
                                                  uint8_t reference_color,
                                                  ARCTransformCandidate *candidate)
{
    int have_offset = 0;
    int dx = 0;
    int dy = 0;
    int selector_used = selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    int output_selector_used =
        output_selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output)) {
            return 0;
        }
        ARCScene input_scene;
        ARCScene output_scene;
        ARCObject input_object;
        ARCObject output_object;
        if (!arc_extract_scene(&task->train[pair].input, background, &input_scene) ||
            !arc_extract_scene(&task->train[pair].output, background, &output_scene) ||
            !arc_select_scene_object(&input_scene,
                                     selector_kind,
                                     input_color,
                                     reference_color,
                                     &input_object) ||
            !arc_select_scene_object(&output_scene,
                                     output_selector_kind,
                                     output_color,
                                     reference_color,
                                     &output_object) ||
            !arc_objects_same_shape(&input_object, &output_object)) {
            return 0;
        }
        if (selector_kind != ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            arc_scene_count_color_objects(&input_scene, input_color) > 1u) {
            selector_used = 1;
        }
        if (output_selector_kind != ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            arc_scene_count_color_objects(&output_scene, output_color) > 1u) {
            output_selector_used = 1;
        }
        int pair_dx = (int)output_object.min_x - (int)input_object.min_x;
        int pair_dy = (int)output_object.min_y - (int)input_object.min_y;
        if (pair_dx == 0 && pair_dy == 0) {
            return 0;
        }
        if (!have_offset) {
            dx = pair_dx;
            dy = pair_dy;
            have_offset = 1;
        } else if (dx != pair_dx || dy != pair_dy) {
            return 0;
        }
    }
    if (!have_offset || !selector_used || !output_selector_used) {
        return 0;
    }

    arc_init_candidate(candidate, ARC_RULE_OBJECT_TRANSLATE, "");
    candidate->input_color = input_color;
    candidate->output_color = output_color;
    candidate->background_color = background;
    candidate->selector_kind = selector_kind;
    candidate->output_selector_kind = output_selector_kind;
    candidate->reference_color = reference_color;
    candidate->dx = dx;
    candidate->dy = dy;
    if (selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
        output_selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_translate_bg_%u_color_%u_to_%u_%+d_%+d",
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 dx,
                 dy);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_translate_%s_to_%s_ref_%u_bg_%u_color_%u_to_%u_%+d_%+d",
                 arc_object_selector_short_name(selector_kind),
                 arc_object_selector_short_name(output_selector_kind),
                 (unsigned)reference_color,
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 dx,
                 dy);
    }
    return 1;
}

static int arc_scene_has_aligned_color_pair(const ARCScene *scene, uint8_t color, int mode)
{
    for (size_t i = 0; i < scene->object_count; ++i) {
        const ARCObject *lhs = &scene->objects[i];
        if (lhs->color != color) {
            continue;
        }
        for (size_t j = i + 1u; j < scene->object_count; ++j) {
            const ARCObject *rhs = &scene->objects[j];
            if (rhs->color != color) {
                continue;
            }
            int horizontal = fabs(lhs->centroid_y - rhs->centroid_y) < 1e-9;
            int vertical = fabs(lhs->centroid_x - rhs->centroid_x) < 1e-9;
            if ((mode == ARC_OBJECT_CONNECT_HORIZONTAL && horizontal) ||
                (mode == ARC_OBJECT_CONNECT_VERTICAL && vertical) ||
                (mode == ARC_OBJECT_CONNECT_BOTH && (horizontal || vertical))) {
                return 1;
            }
        }
    }
    return 0;
}

static int arc_apply_object_connect_candidate(const ARCTransformCandidate *candidate,
                                              const ARCGrid *input,
                                              ARCGrid *output)
{
    ARCScene scene;
    if (!arc_extract_scene(input, candidate->background_color, &scene) ||
        !arc_scene_has_aligned_color_pair(&scene, candidate->input_color, candidate->dx)) {
        return 0;
    }

    if (arc_candidate_render_mode(candidate) == ARC_OBJECT_RENDER_ONLY) {
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = candidate->background_color;
        }
    } else {
        *output = *input;
    }
    if (candidate->dx == ARC_OBJECT_CONNECT_HORIZONTAL ||
        candidate->dx == ARC_OBJECT_CONNECT_BOTH) {
        for (size_t y = 0; y < input->height; ++y) {
            int found = 0;
            size_t min_x = input->width;
            size_t max_x = 0;
            for (size_t x = 0; x < input->width; ++x) {
                if (input->cells[y * input->width + x] != candidate->input_color) {
                    continue;
                }
                if (!found || x < min_x) {
                    min_x = x;
                }
                if (!found || x > max_x) {
                    max_x = x;
                }
                found = 1;
            }
            if (!found || min_x == max_x) {
                continue;
            }
            for (size_t x = min_x; x <= max_x; ++x) {
                output->cells[y * output->width + x] = candidate->output_color;
            }
        }
    }
    if (candidate->dx == ARC_OBJECT_CONNECT_VERTICAL ||
        candidate->dx == ARC_OBJECT_CONNECT_BOTH) {
        for (size_t x = 0; x < input->width; ++x) {
            int found = 0;
            size_t min_y = input->height;
            size_t max_y = 0;
            for (size_t y = 0; y < input->height; ++y) {
                if (input->cells[y * input->width + x] != candidate->input_color) {
                    continue;
                }
                if (!found || y < min_y) {
                    min_y = y;
                }
                if (!found || y > max_y) {
                    max_y = y;
                }
                found = 1;
            }
            if (!found || min_y == max_y) {
                continue;
            }
            for (size_t y = min_y; y <= max_y; ++y) {
                output->cells[y * output->width + x] = candidate->output_color;
            }
        }
    }
    return 1;
}

static int arc_build_object_connect_candidate(const ARCTask *task,
                                              uint8_t background,
                                              uint8_t input_color,
                                              uint8_t output_color,
                                              int mode,
                                              uint8_t render_mode,
                                              ARCTransformCandidate *candidate)
{
    const char *mode_name = mode == ARC_OBJECT_CONNECT_HORIZONTAL
                                ? "horizontal"
                                : mode == ARC_OBJECT_CONNECT_VERTICAL ? "vertical" : "both";
    arc_init_candidate(candidate, ARC_RULE_OBJECT_CONNECT_ALIGNED, "");
    candidate->input_color = input_color;
    candidate->output_color = output_color;
    candidate->background_color = background;
    candidate->dx = mode;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->render_mode = render_mode;
    if (render_mode == ARC_OBJECT_RENDER_ONLY) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_connect_bg_%u_color_%u_to_%u_%s_line_only",
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 mode_name);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_connect_bg_%u_color_%u_to_%u_%s",
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 mode_name);
    }

    int changed_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        ARCGrid produced;
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output) ||
            !arc_apply_object_connect_candidate(candidate,
                                                &task->train[pair].input,
                                                &produced) ||
            !arc_grid_equal(&produced, &task->train[pair].output)) {
            return 0;
        }
        changed_any = changed_any || !arc_grid_equal(&task->train[pair].input, &produced);
    }
    return changed_any;
}

static int arc_apply_object_fill_enclosed_candidate(const ARCTransformCandidate *candidate,
                                                    const ARCGrid *input,
                                                    ARCGrid *output)
{
    uint8_t reachable[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    if (input->width == 0 || input->height == 0) {
        return 0;
    }
    memset(reachable, 0, sizeof(reachable));
    *output = *input;

    size_t head = 0;
    size_t tail = 0;
    for (size_t x = 0; x < input->width; ++x) {
        size_t top = x;
        size_t bottom = (input->height - 1u) * input->width + x;
        if (input->cells[top] != candidate->input_color && !reachable[top]) {
            reachable[top] = 1;
            queue[tail++] = top;
        }
        if (input->cells[bottom] != candidate->input_color && !reachable[bottom]) {
            reachable[bottom] = 1;
            queue[tail++] = bottom;
        }
    }
    for (size_t y = 0; y < input->height; ++y) {
        size_t left = y * input->width;
        size_t right = y * input->width + input->width - 1u;
        if (input->cells[left] != candidate->input_color && !reachable[left]) {
            reachable[left] = 1;
            queue[tail++] = left;
        }
        if (input->cells[right] != candidate->input_color && !reachable[right]) {
            reachable[right] = 1;
            queue[tail++] = right;
        }
    }

    while (head < tail) {
        size_t cell = queue[head++];
        size_t x = cell % input->width;
        size_t y = cell / input->width;
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
            int nx = (int)x + offsets[i][0];
            int ny = (int)y + offsets[i][1];
            if (nx < 0 || ny < 0 ||
                nx >= (int)input->width || ny >= (int)input->height) {
                continue;
            }
            size_t next = (size_t)ny * input->width + (size_t)nx;
            if (reachable[next] || input->cells[next] == candidate->input_color) {
                continue;
            }
            reachable[next] = 1;
            queue[tail++] = next;
        }
    }

    int changed = 0;
    for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
        if (input->cells[cell] == candidate->input_color || reachable[cell]) {
            continue;
        }
        output->cells[cell] = candidate->output_color;
        changed = 1;
    }
    return changed;
}

static int arc_build_object_fill_enclosed_candidate(const ARCTask *task,
                                                    uint8_t border_color,
                                                    uint8_t fill_color,
                                                    ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_OBJECT_FILL_ENCLOSED, "");
    candidate->input_color = border_color;
    candidate->output_color = fill_color;
    candidate->background_color = 0;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->render_mode = ARC_OBJECT_RENDER_OVERLAY;
    snprintf(candidate->name,
             sizeof(candidate->name),
             "object_fill_enclosed_%u_to_%u",
             (unsigned)border_color,
             (unsigned)fill_color);

    int changed_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        ARCGrid produced;
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output) ||
            !arc_apply_object_fill_enclosed_candidate(candidate,
                                                      &task->train[pair].input,
                                                      &produced) ||
            !arc_grid_equal(&produced, &task->train[pair].output)) {
            return 0;
        }
        changed_any = changed_any || !arc_grid_equal(&task->train[pair].input, &produced);
    }
    return changed_any;
}

static int arc_build_selected_object_candidate(const ARCTask *task,
                                               ARCTransformRule op,
                                               uint8_t background,
                                               uint8_t input_color,
                                               uint8_t output_color,
                                               uint8_t selector_kind,
                                               uint8_t reference_color,
                                               ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, op, "");
    candidate->input_color = input_color;
    candidate->output_color = output_color;
    candidate->background_color = background;
    candidate->selector_kind = selector_kind;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->reference_color = reference_color;
    candidate->render_mode = ARC_OBJECT_RENDER_ONLY;

    const char *op_name = op == ARC_RULE_OBJECT_MASK_LARGEST
                              ? "mask"
                              : op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR ? "crop_recolor"
                                                                            : "crop";
    int selector_used = selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    int changed_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        ARCScene scene;
        ARCGrid produced;
        if (!task->train[pair].has_output ||
            !arc_extract_scene(&task->train[pair].input, background, &scene) ||
            !arc_apply_candidate(candidate, &task->train[pair].input, &produced) ||
            !arc_grid_equal(&produced, &task->train[pair].output)) {
            return 0;
        }
        if (selector_kind != ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            arc_scene_count_color_objects(&scene, input_color) > 1u) {
            selector_used = 1;
        }
        changed_any = changed_any || !arc_grid_equal(&task->train[pair].input, &produced);
    }
    if (!selector_used || !changed_any) {
        return 0;
    }

    if (selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_select_%s_%s_ref_%u_bg_%u_color_%u_to_%u",
                 op_name,
                 arc_object_selector_short_name(selector_kind),
                 (unsigned)reference_color,
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_select_%s_%s_bg_%u_color_%u_to_%u",
                 op_name,
                 arc_object_selector_short_name(selector_kind),
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color);
    }
    return 1;
}

static int arc_build_object_relation_candidate(const ARCTask *task,
                                               size_t relation_kind,
                                               uint8_t background,
                                               uint8_t input_color,
                                               uint8_t output_color,
                                               uint8_t selector_kind,
                                               uint8_t output_selector_kind,
                                               uint8_t reference_color,
                                               uint8_t render_mode,
                                               int mode,
                                               ARCTransformCandidate *candidate)
{
    ARCTransformCandidate induced;
    memset(&induced, 0, sizeof(induced));

    switch (relation_kind) {
    case ARC_OBJECT_RELATION_TRANSLATE:
        if (!arc_build_object_translation_candidate(task,
                                                    background,
                                                    input_color,
                                                    output_color,
                                                    selector_kind,
                                                    output_selector_kind,
                                                    reference_color,
                                                    &induced)) {
            return 0;
        }
        if (selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            output_selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR) {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_bg_%u_color_%u_to_%u_%+d_%+d",
                     arc_object_relation_kind_name(relation_kind),
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     induced.dx,
                     induced.dy);
        } else {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_%s_to_%s_ref_%u_bg_%u_color_%u_to_%u_%+d_%+d",
                     arc_object_relation_kind_name(relation_kind),
                     arc_object_selector_short_name(selector_kind),
                     arc_object_selector_short_name(output_selector_kind),
                     (unsigned)reference_color,
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     induced.dx,
                     induced.dy);
        }
        break;
    case ARC_OBJECT_RELATION_CONNECT_ALIGNED: {
        if (!arc_build_object_connect_candidate(task,
                                                background,
                                                input_color,
                                                output_color,
                                                mode,
                                                render_mode,
                                                &induced)) {
            return 0;
        }
        const char *mode_name =
            mode == ARC_OBJECT_CONNECT_HORIZONTAL
                ? "horizontal"
                : mode == ARC_OBJECT_CONNECT_VERTICAL ? "vertical" : "both";
        if (render_mode == ARC_OBJECT_RENDER_ONLY) {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_bg_%u_color_%u_to_%u_%s_line_only",
                     arc_object_relation_kind_name(relation_kind),
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     mode_name);
        } else {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_bg_%u_color_%u_to_%u_%s",
                     arc_object_relation_kind_name(relation_kind),
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     mode_name);
        }
        break;
    }
    case ARC_OBJECT_RELATION_FILL_ENCLOSED:
        if (!arc_build_object_fill_enclosed_candidate(task,
                                                      input_color,
                                                      output_color,
                                                      &induced)) {
            return 0;
        }
        snprintf(induced.name,
                 sizeof(induced.name),
                 "object_relation_%s_%u_to_%u",
                 arc_object_relation_kind_name(relation_kind),
                 (unsigned)induced.input_color,
                 (unsigned)induced.output_color);
        break;
    default:
        return 0;
    }

    induced.op = ARC_RULE_OBJECT_RELATION_INDUCED;
    induced.scale = relation_kind;
    *candidate = induced;
    return 1;
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

    uint8_t relation_backgrounds[2];
    size_t relation_background_count =
        arc_task_relation_backgrounds(task,
                                      relation_backgrounds,
                                      sizeof(relation_backgrounds) /
                                          sizeof(relation_backgrounds[0]));
    static const uint8_t object_selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t background_index = 0;
         background_index < relation_background_count;
         ++background_index) {
        uint8_t background = relation_backgrounds[background_index];
        for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
            if (input_color == background) {
                continue;
            }
            for (size_t selector_index = 0;
                 selector_index < sizeof(object_selectors) / sizeof(object_selectors[0]);
                 ++selector_index) {
                uint8_t selector = object_selectors[selector_index];
                if (arc_build_selected_object_candidate(task,
                                                        ARC_RULE_OBJECT_CROP_LARGEST,
                                                        background,
                                                        input_color,
                                                        input_color,
                                                        selector,
                                                        0,
                                                        &candidate)) {
                    arc_add_candidate(candidates, capacity, &count, &candidate);
                }
                for (uint8_t output_color = 1;
                     output_color < ARC_COLOR_COUNT;
                     ++output_color) {
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            selector,
                                                            0,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_MASK_LARGEST,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            selector,
                                                            0,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                }
            }
            for (uint8_t reference_color = 0;
                 reference_color < ARC_COLOR_COUNT;
                 ++reference_color) {
                if (reference_color == background ||
                    reference_color == input_color) {
                    continue;
                }
                if (arc_build_selected_object_candidate(task,
                                                        ARC_RULE_OBJECT_CROP_LARGEST,
                                                        background,
                                                        input_color,
                                                        input_color,
                                                        ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                        reference_color,
                                                        &candidate)) {
                    arc_add_candidate(candidates, capacity, &count, &candidate);
                }
                for (uint8_t output_color = 1;
                     output_color < ARC_COLOR_COUNT;
                     ++output_color) {
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                            reference_color,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_MASK_LARGEST,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                            reference_color,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                }
            }
        }
    }
    static const uint8_t translate_selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    static const uint8_t translate_output_selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t background_index = 0;
         background_index < relation_background_count;
         ++background_index) {
        uint8_t background = relation_backgrounds[background_index];
        for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
            if (input_color == background) {
                continue;
            }
            for (uint8_t output_color = 0; output_color < ARC_COLOR_COUNT; ++output_color) {
                if (output_color == background) {
                    continue;
                }
                for (size_t selector_index = 0;
                     selector_index < sizeof(translate_selectors) /
                                          sizeof(translate_selectors[0]);
                     ++selector_index) {
                    for (size_t output_selector_index = 0;
                         output_selector_index < sizeof(translate_output_selectors) /
                                                     sizeof(translate_output_selectors[0]);
                         ++output_selector_index) {
                        if (arc_build_object_relation_candidate(
                                task,
                                ARC_OBJECT_RELATION_TRANSLATE,
                                background,
                                input_color,
                                output_color,
                                translate_selectors[selector_index],
                                translate_output_selectors[output_selector_index],
                                0,
                                ARC_OBJECT_RENDER_ONLY,
                                0,
                                &candidate)) {
                            arc_add_candidate(candidates, capacity, &count, &candidate);
                        }
                    }
                }
                for (uint8_t reference_color = 0;
                     reference_color < ARC_COLOR_COUNT;
                     ++reference_color) {
                    if (reference_color == background ||
                        reference_color == input_color) {
                        continue;
                    }
                    if (arc_build_object_relation_candidate(task,
                                                            ARC_OBJECT_RELATION_TRANSLATE,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                            ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
                                                            reference_color,
                                                            ARC_OBJECT_RENDER_ONLY,
                                                            0,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                }
                for (int mode = ARC_OBJECT_CONNECT_HORIZONTAL;
                     mode <= ARC_OBJECT_CONNECT_BOTH;
                     ++mode) {
                    for (uint8_t render_mode = ARC_OBJECT_RENDER_OVERLAY;
                         render_mode < ARC_OBJECT_RENDER_KIND_COUNT;
                         ++render_mode) {
                        if (arc_build_object_relation_candidate(task,
                                                                ARC_OBJECT_RELATION_CONNECT_ALIGNED,
                                                                background,
                                                                input_color,
                                                                output_color,
                                                                ARC_OBJECT_SELECTOR_ALL_COLOR,
                                                                ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
                                                                0,
                                                                render_mode,
                                                                mode,
                                                                &candidate)) {
                            arc_add_candidate(candidates, capacity, &count, &candidate);
                        }
                    }
                }
            }
        }
    }

    for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
        for (uint8_t fill_color = 1; fill_color < ARC_COLOR_COUNT; ++fill_color) {
            if (arc_build_object_relation_candidate(task,
                                                    ARC_OBJECT_RELATION_FILL_ENCLOSED,
                                                    0,
                                                    color,
                                                    fill_color,
                                                    ARC_OBJECT_SELECTOR_ALL_COLOR,
                                                    ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
                                                    0,
                                                    ARC_OBJECT_RENDER_OVERLAY,
                                                    0,
                                                    &candidate)) {
                arc_add_candidate(candidates, capacity, &count, &candidate);
            }
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

static uint64_t arc_scene_hash_mix(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}

static uint64_t arc_shape_hash_from_mask(const ARCGrid *grid,
                                         const uint8_t *mask,
                                         size_t min_x,
                                         size_t min_y,
                                         size_t max_x,
                                         size_t max_y,
                                         size_t count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t width = max_x >= min_x ? max_x - min_x + 1u : 0u;
    size_t height = max_y >= min_y ? max_y - min_y + 1u : 0u;
    hash = arc_scene_hash_mix(hash, width);
    hash = arc_scene_hash_mix(hash, height);
    hash = arc_scene_hash_mix(hash, count);
    for (size_t y = min_y; y <= max_y && y < grid->height; ++y) {
        for (size_t x = min_x; x <= max_x && x < grid->width; ++x) {
            size_t cell = y * grid->width + x;
            hash = arc_scene_hash_mix(hash, mask[cell] ? 1u : 0u);
        }
    }
    return hash;
}

static void arc_scene_clear(ARCScene *scene, uint8_t background)
{
    memset(scene, 0, sizeof(*scene));
    scene->background_color = background;
}

static int arc_ranges_overlap(size_t lhs_min,
                              size_t lhs_max,
                              size_t rhs_min,
                              size_t rhs_max)
{
    return lhs_min <= rhs_max && rhs_min <= lhs_max;
}

static int arc_objects_adjacent(const ARCObject *lhs, const ARCObject *rhs)
{
    int horizontal_touch =
        (lhs->max_x + 1u == rhs->min_x || rhs->max_x + 1u == lhs->min_x) &&
        arc_ranges_overlap(lhs->min_y, lhs->max_y, rhs->min_y, rhs->max_y);
    int vertical_touch =
        (lhs->max_y + 1u == rhs->min_y || rhs->max_y + 1u == lhs->min_y) &&
        arc_ranges_overlap(lhs->min_x, lhs->max_x, rhs->min_x, rhs->max_x);
    return horizontal_touch || vertical_touch;
}

static int arc_objects_aligned(const ARCObject *lhs, const ARCObject *rhs)
{
    return lhs->min_x == rhs->min_x ||
           lhs->max_x == rhs->max_x ||
           lhs->min_y == rhs->min_y ||
           lhs->max_y == rhs->max_y ||
           fabs(lhs->centroid_x - rhs->centroid_x) < 1e-9 ||
           fabs(lhs->centroid_y - rhs->centroid_y) < 1e-9;
}

static void arc_scene_add_relation_evidence(ARCScene *scene)
{
    for (size_t i = 0; i < scene->object_count; ++i) {
        for (size_t j = i + 1u; j < scene->object_count; ++j) {
            const ARCObject *lhs = &scene->objects[i];
            const ARCObject *rhs = &scene->objects[j];
            if (lhs->count == rhs->count &&
                lhs->shape_hash == rhs->shape_hash) {
                ++scene->repeated_shape_pairs;
            }
            if (arc_objects_aligned(lhs, rhs)) {
                ++scene->aligned_pairs;
            }
            if (arc_objects_adjacent(lhs, rhs)) {
                ++scene->adjacent_pairs;
            }
        }
    }
}

static int arc_extract_scene(const ARCGrid *grid, uint8_t background, ARCScene *scene)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    uint8_t current_mask[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    memset(visited, 0, sizeof(visited));
    arc_scene_clear(scene, background);

    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        uint8_t color = grid->cells[start];
        if (visited[start] || color == background) {
            continue;
        }
        if (scene->object_count >= ARC_MAX_SCENE_OBJECTS) {
            return 0;
        }

        memset(current_mask, 0, sizeof(current_mask));
        size_t head = 0;
        size_t tail = 0;
        size_t count = 0;
        size_t min_x = grid->width;
        size_t min_y = grid->height;
        size_t max_x = 0;
        size_t max_y = 0;
        double sum_x = 0.0;
        double sum_y = 0.0;

        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            current_mask[cell] = 1;
            ++count;
            sum_x += (double)x;
            sum_y += (double)y;
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
            for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
                int nx = (int)x + offsets[i][0];
                int ny = (int)y + offsets[i][1];
                if (nx < 0 || ny < 0 ||
                    nx >= (int)grid->width || ny >= (int)grid->height) {
                    continue;
                }
                size_t next = (size_t)ny * grid->width + (size_t)nx;
                if (visited[next] || grid->cells[next] != color) {
                    continue;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }

        ARCObject *object = &scene->objects[scene->object_count];
        object->id = scene->object_count;
        object->color = color;
        object->count = count;
        object->min_x = min_x;
        object->min_y = min_y;
        object->max_x = max_x;
        object->max_y = max_y;
        object->centroid_x = count == 0 ? 0.0 : sum_x / (double)count;
        object->centroid_y = count == 0 ? 0.0 : sum_y / (double)count;
        object->shape_hash = arc_shape_hash_from_mask(grid,
                                                       current_mask,
                                                       min_x,
                                                       min_y,
                                                       max_x,
                                                       max_y,
                                                       count);
        object->touches_border = min_x == 0 || min_y == 0 ||
                                 max_x + 1u == grid->width ||
                                 max_y + 1u == grid->height;
        ++scene->object_count;
    }

    arc_scene_add_relation_evidence(scene);
    return 1;
}

static int arc_scene_unique_color_object(const ARCScene *scene,
                                         uint8_t color,
                                         ARCObject *object_out)
{
    int found = 0;
    ARCObject object;
    memset(&object, 0, sizeof(object));
    for (size_t i = 0; i < scene->object_count; ++i) {
        if (scene->objects[i].color != color) {
            continue;
        }
        if (found) {
            return 0;
        }
        object = scene->objects[i];
        found = 1;
    }
    if (!found) {
        return 0;
    }
    if (object_out != NULL) {
        *object_out = object;
    }
    return 1;
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
    ARCScene input_scene;
    ARCScene output_scene;
    int have_input_scene = arc_extract_scene(input, 0, &input_scene);
    int have_output_scene = arc_extract_scene(output, 0, &output_scene);

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
    if (have_input_scene) {
        features->input_object_count +=
            (double)input_scene.object_count / (double)ARC_MAX_SCENE_OBJECTS;
        features->input_repeated_shape_pairs +=
            (double)input_scene.repeated_shape_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->input_aligned_pairs +=
            (double)input_scene.aligned_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->input_adjacent_pairs +=
            (double)input_scene.adjacent_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
    }
    if (have_output_scene) {
        features->output_object_count +=
            (double)output_scene.object_count / (double)ARC_MAX_SCENE_OBJECTS;
        features->output_repeated_shape_pairs +=
            (double)output_scene.repeated_shape_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->output_aligned_pairs +=
            (double)output_scene.aligned_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->output_adjacent_pairs +=
            (double)output_scene.adjacent_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
    }
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
    ARC_NORMALIZE_TASK_FEATURE(input_object_count);
    ARC_NORMALIZE_TASK_FEATURE(output_object_count);
    ARC_NORMALIZE_TASK_FEATURE(input_repeated_shape_pairs);
    ARC_NORMALIZE_TASK_FEATURE(output_repeated_shape_pairs);
    ARC_NORMALIZE_TASK_FEATURE(input_aligned_pairs);
    ARC_NORMALIZE_TASK_FEATURE(output_aligned_pairs);
    ARC_NORMALIZE_TASK_FEATURE(input_adjacent_pairs);
    ARC_NORMALIZE_TASK_FEATURE(output_adjacent_pairs);
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
    out[24] = (nn_real)features->input_object_count;
    out[25] = (nn_real)features->output_object_count;
    out[26] = (nn_real)features->input_repeated_shape_pairs;
    out[27] = (nn_real)features->output_repeated_shape_pairs;
    out[28] = (nn_real)features->input_aligned_pairs;
    out[29] = (nn_real)features->output_aligned_pairs;
    out[30] = (nn_real)features->input_adjacent_pairs;
    out[31] = (nn_real)features->output_adjacent_pairs;
}

static int arc_apply_largest_component_candidate(const ARCTransformCandidate *candidate,
                                                 const ARCGrid *input,
                                                 ARCGrid *output)
{
    if (candidate->selector_kind != ARC_OBJECT_SELECTOR_ALL_COLOR) {
        ARCScene scene;
        ARCObject object;
        if (!arc_extract_scene(input, candidate->background_color, &scene) ||
            !arc_select_scene_object(&scene,
                                     arc_candidate_selector_kind(candidate),
                                     candidate->input_color,
                                     candidate->reference_color,
                                     &object)) {
            return 0;
        }

        if (candidate->op == ARC_RULE_OBJECT_MASK_LARGEST) {
            arc_init_grid(output, input->width, input->height);
            for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
                output->cells[cell] = 0;
            }
            for (size_t y = object.min_y; y <= object.max_y; ++y) {
                for (size_t x = object.min_x; x <= object.max_x; ++x) {
                    size_t cell = y * input->width + x;
                    if (input->cells[cell] == candidate->input_color) {
                        output->cells[cell] = candidate->output_color;
                    }
                }
            }
            return 1;
        }

        size_t width = object.max_x - object.min_x + 1u;
        size_t height = object.max_y - object.min_y + 1u;
        arc_init_grid(output, width, height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t source = (object.min_y + y) * input->width + object.min_x + x;
                size_t target = y * width + x;
                if (input->cells[source] != candidate->input_color) {
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

static int arc_apply_object_translate_candidate(const ARCTransformCandidate *candidate,
                                                const ARCGrid *input,
                                                ARCGrid *output)
{
    ARCScene scene;
    ARCObject object;
    if (!arc_extract_scene(input, candidate->background_color, &scene) ||
        !arc_select_scene_object(&scene,
                                 arc_candidate_selector_kind(candidate),
                                 candidate->input_color,
                                 candidate->reference_color,
                                 &object)) {
        return 0;
    }

    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = candidate->background_color;
    }
    for (size_t y = object.min_y; y <= object.max_y; ++y) {
        for (size_t x = object.min_x; x <= object.max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != candidate->input_color) {
                continue;
            }
            int out_x = (int)x + candidate->dx;
            int out_y = (int)y + candidate->dy;
            if (out_x < 0 || out_y < 0 ||
                out_x >= (int)input->width || out_y >= (int)input->height) {
                return 0;
            }
            output->cells[(size_t)out_y * output->width + (size_t)out_x] =
                candidate->output_color;
        }
    }
    return 1;
}

static int arc_apply_candidate(const ARCTransformCandidate *candidate,
                               const ARCGrid *input,
                               ARCGrid *output)
{
    if (arc_candidate_is_composed(candidate)) {
        ARCTransformCandidate first;
        ARCTransformCandidate second;
        ARCGrid intermediate;
        arc_candidate_from_step(&candidate->steps[0], &first);
        arc_candidate_from_step(&candidate->steps[1], &second);
        return arc_apply_candidate(&first, input, &intermediate) &&
               arc_apply_candidate(&second, &intermediate, output);
    }

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
    case ARC_RULE_OBJECT_TRANSLATE:
        return arc_apply_object_translate_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_CONNECT_ALIGNED:
        return arc_apply_object_connect_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_FILL_ENCLOSED:
        return arc_apply_object_fill_enclosed_candidate(candidate, input, output);
    case ARC_RULE_HYPOTHESIS_APPLIED:
        return arc_apply_hypothesis_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_RELATION_INDUCED:
        switch (arc_object_relation_kind(candidate)) {
        case ARC_OBJECT_RELATION_TRANSLATE:
            return arc_apply_object_translate_candidate(candidate, input, output);
        case ARC_OBJECT_RELATION_CONNECT_ALIGNED:
            return arc_apply_object_connect_candidate(candidate, input, output);
        case ARC_OBJECT_RELATION_FILL_ENCLOSED:
            return arc_apply_object_fill_enclosed_candidate(candidate, input, output);
        default:
            return 0;
        }
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
    if (arc_candidate_is_composed(candidate)) {
        return 0;
    }
    return candidate->op == ARC_RULE_OBJECT_CROP_LARGEST ||
           candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR ||
           candidate->op == ARC_RULE_OBJECT_MASK_LARGEST ||
           candidate->op == ARC_RULE_OBJECT_TRANSLATE ||
           candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED ||
           candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED ||
           candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED ||
           candidate->op == ARC_RULE_HYPOTHESIS_APPLIED;
}

static int arc_candidate_is_selected_object_program(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           !arc_candidate_is_composed(candidate) &&
           candidate->selector_kind != ARC_OBJECT_SELECTOR_ALL_COLOR &&
           (candidate->op == ARC_RULE_OBJECT_CROP_LARGEST ||
            candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR ||
            candidate->op == ARC_RULE_OBJECT_MASK_LARGEST);
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
    case ARC_RULE_OBJECT_TRANSLATE:
        return 12;
    case ARC_RULE_OBJECT_CONNECT_ALIGNED:
        return 13;
    case ARC_RULE_OBJECT_FILL_ENCLOSED:
        return 14;
    case ARC_RULE_HYPOTHESIS_APPLIED:
        return 15;
    case ARC_RULE_OBJECT_RELATION_INDUCED:
        return 12;
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
    case ARC_RULE_COMPOSED:
        return 11;
    }
    return 0;
}

static int arc_candidate_is_hypothesis_induced(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           (candidate->op == ARC_RULE_HYPOTHESIS_APPLIED ||
            strncmp(candidate->name, "hypothesis_", strlen("hypothesis_")) == 0);
}

static double arc_structural_program_score(const ARCTransformCandidate *candidate)
{
    if (arc_candidate_is_hypothesis_induced(candidate)) {
        return (double)ARC_HYPOTHESIS_STRUCTURAL_SCORE -
               (double)(abs(candidate->dx) + abs(candidate->dy));
    }
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
    case ARC_RULE_OBJECT_TRANSLATE:
        return 98.0 - (double)(abs(candidate->dx) + abs(candidate->dy));
    case ARC_RULE_OBJECT_CONNECT_ALIGNED:
        return 99.0 - (double)candidate->dx;
    case ARC_RULE_OBJECT_FILL_ENCLOSED:
        return 97.0;
    case ARC_RULE_OBJECT_RELATION_INDUCED:
        switch (arc_object_relation_kind(candidate)) {
        case ARC_OBJECT_RELATION_TRANSLATE:
            return 98.0 - (double)(abs(candidate->dx) + abs(candidate->dy));
        case ARC_OBJECT_RELATION_CONNECT_ALIGNED:
            return 99.0 - (double)candidate->dx;
        case ARC_OBJECT_RELATION_FILL_ENCLOSED:
            return 97.0;
        default:
            return 90.0;
        }
    case ARC_RULE_HYPOTHESIS_APPLIED:
        return (double)ARC_HYPOTHESIS_STRUCTURAL_SCORE -
               (double)(abs(candidate->dx) + abs(candidate->dy));
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
    case ARC_RULE_COMPOSED:
        return -40.0;
    }
    return 0.0;
}

static int arc_candidate_to_program(const ARCTransformCandidate *candidate,
                                    NNProgram *program)
{
    if (candidate == NULL || program == NULL) {
        return 0;
    }

    if (arc_candidate_is_composed(candidate)) {
        ARCTransformCandidate first_candidate;
        ARCTransformCandidate second_candidate;
        NNProgram first_program;
        NNProgram second_program;
        arc_candidate_from_step(&candidate->steps[0], &first_candidate);
        arc_candidate_from_step(&candidate->steps[1], &second_candidate);
        if (!arc_candidate_to_program(&first_candidate, &first_program) ||
            !arc_candidate_to_program(&second_candidate, &second_program) ||
            !nn_program_compose(program, &first_program, &second_program, candidate->name)) {
            return 0;
        }
        nn_program_add_param(program,
                             "first_op",
                             (double)first_candidate.op,
                             (int64_t)first_candidate.op);
        nn_program_add_param(program,
                             "second_op",
                             (double)second_candidate.op,
                             (int64_t)second_candidate.op);
        nn_program_add_param(program,
                             "first_family",
                             (double)arc_candidate_family(&first_candidate),
                             (int64_t)arc_candidate_family(&first_candidate));
        nn_program_add_param(program,
                             "second_family",
                             (double)arc_candidate_family(&second_candidate),
                             (int64_t)arc_candidate_family(&second_candidate));
    } else if (candidate->op == ARC_RULE_HYPOTHESIS_APPLIED) {
        NNHypothesis hypothesis;
        return arc_hypothesis_from_candidate(candidate, &hypothesis) &&
               nn_hypothesis_compile(&hypothesis, program);
    } else if (candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED) {
        NNProgram scene;
        NNProgram relation;
        size_t relation_kind = arc_object_relation_kind(candidate);
        int relation_mode = relation_kind == ARC_OBJECT_RELATION_CONNECT_ALIGNED
                                ? candidate->dx
                                : 0;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&relation,
                        "arc_agi",
                        "object_relation_induced",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_GRID,
                        116);
        nn_program_add_param(&relation,
                             "relation_kind",
                             (double)relation_kind,
                             (int64_t)relation_kind);
        nn_program_add_param(&relation,
                             "selector_kind",
                             (double)arc_candidate_selector_kind(candidate),
                             (int64_t)arc_candidate_selector_kind(candidate));
        nn_program_add_param(&relation,
                             "output_selector_kind",
                             (double)arc_candidate_output_selector_kind(candidate),
                             (int64_t)arc_candidate_output_selector_kind(candidate));
        nn_program_add_param(&relation,
                             "reference_color",
                             (double)candidate->reference_color,
                             candidate->reference_color);
        nn_program_add_param(&relation,
                             "render_mode",
                             (double)arc_candidate_render_mode(candidate),
                             arc_candidate_render_mode(candidate));
        nn_program_add_param(&relation,
                             "input_color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&relation,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        nn_program_add_param(&relation,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_add_param(&relation, "dx", (double)candidate->dx, candidate->dx);
        nn_program_add_param(&relation, "dy", (double)candidate->dy, candidate->dy);
        nn_program_add_param(&relation, "mode", (double)relation_mode, relation_mode);
        nn_program_add_feature(&relation, (nn_real)((double)relation_kind / 4.0));
        nn_program_add_feature(&relation,
                               (nn_real)((double)arc_candidate_selector_kind(candidate) /
                                         (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
        nn_program_add_feature(
            &relation,
            (nn_real)((double)arc_candidate_output_selector_kind(candidate) /
                      (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
        nn_program_add_feature(&relation,
                               (nn_real)((double)arc_candidate_render_mode(candidate) /
                                         (double)(ARC_OBJECT_RENDER_KIND_COUNT - 1)));
        nn_program_add_feature(&relation, (nn_real)((double)candidate->reference_color / 9.0));
        nn_program_add_feature(&relation, (nn_real)((double)candidate->input_color / 9.0));
        nn_program_add_feature(&relation, (nn_real)((double)candidate->output_color / 9.0));
        nn_program_add_feature(&relation,
                               (nn_real)((double)candidate->dx / (double)ARC_MAX_GRID_WIDTH));
        nn_program_add_feature(&relation,
                               (nn_real)((double)candidate->dy / (double)ARC_MAX_GRID_HEIGHT));
        if (!nn_program_compose(program, &scene, &relation, candidate->name)) {
            return 0;
        }
    } else if (candidate->op == ARC_RULE_OBJECT_TRANSLATE) {
        NNProgram scene;
        NNProgram selector;
        NNProgram mover;
        NNProgram renderer;
        NNProgram selected;
        NNProgram moved;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&selector,
                        "arc_agi",
                        "select_object_by_selector",
                        NN_PROGRAM_KIND_SELECTOR,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_OBJECT,
                        111);
        nn_program_add_param(&selector,
                             "color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&selector,
                             "selector_kind",
                             (double)arc_candidate_selector_kind(candidate),
                             arc_candidate_selector_kind(candidate));
        nn_program_add_param(&selector,
                             "reference_color",
                             (double)candidate->reference_color,
                             candidate->reference_color);
        nn_program_init(&mover,
                        "arc_agi",
                        "translate_object",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT,
                        NN_PROGRAM_VALUE_OBJECT,
                        112);
        nn_program_add_param(&mover, "dx", (double)candidate->dx, candidate->dx);
        nn_program_add_param(&mover, "dy", (double)candidate->dy, candidate->dy);
        nn_program_init(&renderer,
                        "arc_agi",
                        "render_object_grid",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT,
                        NN_PROGRAM_VALUE_GRID,
                        113);
        nn_program_add_param(&renderer,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_add_param(&renderer,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        if (!nn_program_compose(&selected,
                                &scene,
                                &selector,
                                "object_scene_then_select_color") ||
            !nn_program_compose(&moved,
                                &selected,
                                &mover,
                                "object_scene_select_then_translate") ||
            !nn_program_compose(program, &moved, &renderer, candidate->name)) {
            return 0;
        }
    } else if (candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED) {
        NNProgram scene;
        NNProgram connector;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&connector,
                        "arc_agi",
                        "connect_aligned_color_objects",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_GRID,
                        114);
        nn_program_add_param(&connector,
                             "color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&connector,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        nn_program_add_param(&connector, "mode", (double)candidate->dx, candidate->dx);
        nn_program_add_param(&connector,
                             "render_mode",
                             (double)arc_candidate_render_mode(candidate),
                             arc_candidate_render_mode(candidate));
        if (!nn_program_compose(program, &scene, &connector, candidate->name)) {
            return 0;
        }
    } else if (candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED) {
        NNProgram scene;
        NNProgram filler;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&filler,
                        "arc_agi",
                        "fill_enclosed_by_color",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_GRID,
                        115);
        nn_program_add_param(&filler,
                             "border_color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&filler,
                             "fill_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        if (!nn_program_compose(program, &scene, &filler, candidate->name)) {
            return 0;
        }
    } else if (arc_candidate_is_object_program(candidate)) {
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
                        candidate->selector_kind == ARC_OBJECT_SELECTOR_ALL_COLOR
                            ? "largest_component"
                            : "select_object_by_selector",
                        NN_PROGRAM_KIND_SELECTOR,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_OBJECT,
                        101);
        nn_program_add_param(&largest,
                             "selector_kind",
                             (double)arc_candidate_selector_kind(candidate),
                             arc_candidate_selector_kind(candidate));
        nn_program_add_param(&largest,
                             "color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&largest,
                             "reference_color",
                             (double)candidate->reference_color,
                             candidate->reference_color);
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
    nn_program_add_param(program,
                         "selector_kind",
                         (double)arc_candidate_selector_kind(candidate),
                         arc_candidate_selector_kind(candidate));
    nn_program_add_param(program,
                         "output_selector_kind",
                         (double)arc_candidate_output_selector_kind(candidate),
                         arc_candidate_output_selector_kind(candidate));
    nn_program_add_param(program,
                         "reference_color",
                         (double)candidate->reference_color,
                         candidate->reference_color);
    nn_program_add_param(program,
                         "render_mode",
                         (double)arc_candidate_render_mode(candidate),
                         arc_candidate_render_mode(candidate));
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
    nn_program_add_feature(program, (nn_real)((double)candidate->input_color / 9.0));
    nn_program_add_feature(program,
                           (nn_real)((double)candidate->dx / (double)ARC_MAX_GRID_WIDTH));
    nn_program_add_feature(program,
                           (nn_real)((double)candidate->dy / (double)ARC_MAX_GRID_HEIGHT));
    nn_program_add_feature(program, (nn_real)(arc_candidate_is_object_relation(candidate) ? 1.0 : 0.0));
    nn_program_add_feature(program,
                           (nn_real)((double)arc_candidate_selector_kind(candidate) /
                                     (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
    nn_program_add_feature(
        program,
        (nn_real)((double)arc_candidate_output_selector_kind(candidate) /
                  (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
    nn_program_add_feature(program,
                           (nn_real)((double)arc_candidate_render_mode(candidate) /
                                     (double)(ARC_OBJECT_RENDER_KIND_COUNT - 1)));
    nn_program_add_feature(program, (nn_real)((double)candidate->reference_color / 9.0));
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

static NeuralNet *create_arc_hypothesis_prior_model(unsigned int seed)
{
    static const size_t layers[] = {ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT, 24, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed ^ 0x9e43a71bu;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int arc_program_learned_init(ARCProgramLearnedState *state, unsigned int seed)
{
    memset(state, 0, sizeof(*state));
    state->net = create_arc_program_model(seed);
    state->hypothesis_prior_net = create_arc_hypothesis_prior_model(seed);
    if (state->net == NULL || state->hypothesis_prior_net == NULL) {
        nn_free(state->net);
        nn_free(state->hypothesis_prior_net);
        memset(state, 0, sizeof(*state));
        return 0;
    }
    return 1;
}

static void arc_program_learned_free(ARCProgramLearnedState *state)
{
    if (state == NULL) {
        return;
    }
    nn_free(state->net);
    nn_free(state->hypothesis_prior_net);
    memset(state, 0, sizeof(*state));
}

static size_t arc_hypothesis_template_bucket(uint32_t template_id)
{
    switch (template_id) {
    case 1024:
        return 0;
    case 1025:
        return 1;
    default:
        return (size_t)(template_id % ARC_HYPOTHESIS_TEMPLATE_BUCKETS);
    }
}

static size_t arc_hypothesis_slot_type_bucket(NNHypothesisSlotType type)
{
    switch (type) {
    case NN_HYPOTHESIS_SLOT_INTEGER:
    case NN_HYPOTHESIS_SLOT_COLOR:
    case NN_HYPOTHESIS_SLOT_OFFSET:
    case NN_HYPOTHESIS_SLOT_SELECTOR_KIND:
    case NN_HYPOTHESIS_SLOT_RELATION_KIND:
    case NN_HYPOTHESIS_SLOT_RENDER_MODE:
    case NN_HYPOTHESIS_SLOT_REFERENCE_COLOR:
    case NN_HYPOTHESIS_SLOT_SCALAR:
    case NN_HYPOTHESIS_SLOT_BOOLEAN:
        return (size_t)type;
    case NN_HYPOTHESIS_SLOT_CUSTOM:
    default:
        return 0;
    }
}

static uint64_t arc_hypothesis_prior_mix(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}

static size_t arc_hypothesis_slot_value_bucket(const NNHypothesis *hypothesis,
                                               const NNHypothesisSlotValue *slot)
{
    uint64_t hash = hypothesis != NULL && hypothesis->template_ref != NULL
                        ? hypothesis->template_ref->template_id
                        : 0;
    hash = arc_hypothesis_prior_mix(hash, (uint64_t)slot->type);
    hash = arc_hypothesis_prior_mix(hash, (uint64_t)slot->integer);
    for (const unsigned char *p = (const unsigned char *)slot->name; *p != '\0'; ++p) {
        hash = arc_hypothesis_prior_mix(hash, *p);
    }
    return (size_t)(hash % ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS);
}

static void arc_hypothesis_prior_observe(ARCProgramLearnedState *state,
                                         const NNHypothesis *hypothesis,
                                         int certified)
{
    if (state == NULL || hypothesis == NULL || hypothesis->template_ref == NULL) {
        return;
    }
    size_t template_bucket =
        arc_hypothesis_template_bucket(hypothesis->template_ref->template_id);
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        if (!slot->bound) {
            continue;
        }
        size_t type_bucket = arc_hypothesis_slot_type_bucket(slot->type);
        size_t value_bucket = arc_hypothesis_slot_value_bucket(hypothesis, slot);
        ++state->hypothesis_slot_proposal_count[template_bucket]
                                                [type_bucket]
                                                [value_bucket];
        if (certified) {
            ++state->hypothesis_slot_certified_count[template_bucket]
                                                   [type_bucket]
                                                   [value_bucket];
        }
    }
}

static double arc_hypothesis_slot_log_odds(const ARCProgramLearnedState *state,
                                           const NNHypothesis *hypothesis,
                                           size_t *proposal_count_out)
{
    if (proposal_count_out != NULL) {
        *proposal_count_out = 0;
    }
    if (state == NULL || hypothesis == NULL || hypothesis->template_ref == NULL) {
        return 0.0;
    }
    size_t template_bucket =
        arc_hypothesis_template_bucket(hypothesis->template_ref->template_id);
    double score = 0.0;
    size_t proposal_count = 0;
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        if (!slot->bound) {
            continue;
        }
        size_t type_bucket = arc_hypothesis_slot_type_bucket(slot->type);
        size_t value_bucket = arc_hypothesis_slot_value_bucket(hypothesis, slot);
        size_t proposals = state->hypothesis_slot_proposal_count[template_bucket]
                                                            [type_bucket]
                                                            [value_bucket];
        size_t certified = state->hypothesis_slot_certified_count[template_bucket]
                                                            [type_bucket]
                                                            [value_bucket];
        if (proposals == 0) {
            continue;
        }
        proposal_count += proposals;
        double positives = (double)certified + 1.0;
        double negatives = (double)(proposals - certified) + 1.0;
        score += log(positives / negatives);
    }
    if (proposal_count_out != NULL) {
        *proposal_count_out = proposal_count;
    }
    return score;
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
    if (arc_candidate_is_object_relation(candidate)) {
        return 20.0 * state->family_score[family];
    }
    if (arc_candidate_is_selected_object_program(candidate)) {
        return 10.0 * state->family_score[family];
    }
    return state->family_score[family];
}

static double arc_learned_relation_score(const ARCProgramLearnedState *state,
                                         const ARCTransformCandidate *candidate)
{
    size_t relation_kind = arc_object_relation_kind(candidate);
    if (state == NULL ||
        relation_kind == 0 ||
        relation_kind >= ARC_OBJECT_RELATION_KIND_COUNT ||
        candidate->input_color >= ARC_COLOR_COUNT ||
        candidate->output_color >= ARC_COLOR_COUNT ||
        arc_candidate_selector_kind(candidate) >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        arc_candidate_output_selector_kind(candidate) >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        arc_candidate_render_mode(candidate) >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return 0.0;
    }
    return 2.0 * state->relation_score[relation_kind] +
           state->relation_color_score[relation_kind]
                                      [candidate->input_color]
                                      [candidate->output_color] +
           0.5 * state->relation_background_score[relation_kind]
                                                [candidate->background_color] +
           0.5 * state->relation_selector_score[relation_kind]
                                              [arc_candidate_selector_kind(candidate)] +
           0.25 * state->relation_output_selector_score[relation_kind]
                                                     [arc_candidate_output_selector_kind(candidate)] +
           0.25 * state->relation_render_score[relation_kind]
                                            [arc_candidate_render_mode(candidate)];
}

static double arc_learned_object_score(const ARCProgramLearnedState *state,
                                       const ARCTransformCandidate *candidate)
{
    if (state == NULL ||
        !arc_candidate_is_selected_object_program(candidate) ||
        arc_candidate_selector_kind(candidate) >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        candidate->input_color >= ARC_COLOR_COUNT ||
        candidate->output_color >= ARC_COLOR_COUNT ||
        arc_candidate_render_mode(candidate) >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return 0.0;
    }
    return 40.0 * state->object_selector_score[arc_candidate_selector_kind(candidate)] +
           20.0 * state->object_color_score[candidate->input_color][candidate->output_color] +
           10.0 * state->object_render_score[arc_candidate_render_mode(candidate)];
}

static double arc_learned_program_score(ARCProgramLearnedState *state,
                                        const ARCTransformCandidate *candidate,
                                        const ARCProgramTaskFeatures *task_features)
{
    if (state == NULL || state->net == NULL || state->trained_tasks == 0) {
        return arc_learned_family_score(state, candidate) +
               arc_learned_relation_score(state, candidate) +
               arc_learned_object_score(state, candidate);
    }
    if (arc_candidate_is_composed(candidate)) {
        ARCTransformCandidate first;
        ARCTransformCandidate second;
        arc_candidate_from_step(&candidate->steps[0], &first);
        arc_candidate_from_step(&candidate->steps[1], &second);
        double first_score = arc_learned_program_score(state, &first, task_features);
        double second_score = arc_learned_program_score(state, &second, task_features);
        return -40.0 + 0.5 * (first_score + second_score);
    }
    nn_real features[ARC_PROGRAM_FEATURE_COUNT];
    nn_real output[1];
    if (!arc_encode_program_features(candidate,
                                     task_features,
                                     features,
                                     ARC_PROGRAM_FEATURE_COUNT)) {
        return arc_learned_family_score(state, candidate) +
               arc_learned_relation_score(state, candidate) +
               arc_learned_object_score(state, candidate);
    }
    nn_forward(state->net, features, output);
    return (double)output[0] +
           arc_learned_relation_score(state, candidate) +
           arc_learned_object_score(state, candidate);
}

static int arc_program_eval_is_exact_success(const ARCProgramEval *eval)
{
    return eval->train_consistent &&
           eval->test_supported &&
           eval->test_cells > 0 &&
           eval->test_correct_cells == eval->test_cells;
}

static int arc_program_learned_train_hypothesis_prior(ARCProgramLearnedState *state,
                                                      const ARCTransformCandidate *candidates,
                                                      const ARCProgramEval *evals,
                                                      size_t candidate_count,
                                                      nn_real learning_rate)
{
    if (state == NULL || state->hypothesis_prior_net == NULL ||
        candidates == NULL || evals == NULL || candidate_count == 0 ||
        learning_rate <= (nn_real)0) {
        return 0;
    }

    size_t sample_count = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        sample_count += arc_candidate_is_hypothesis_induced(&candidates[i]) ? 1u : 0u;
    }
    if (sample_count == 0) {
        return 1;
    }

    nn_real *features =
        calloc(sample_count * ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT, sizeof(*features));
    nn_real *targets = calloc(sample_count, sizeof(*targets));
    if (features == NULL || targets == NULL) {
        free(features);
        free(targets);
        return 0;
    }

    size_t row = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!arc_candidate_is_hypothesis_induced(&candidates[i])) {
            continue;
        }
        NNHypothesis hypothesis;
        size_t encoded_count = 0;
        if (!arc_hypothesis_from_candidate(&candidates[i], &hypothesis) ||
            !nn_hypothesis_encode_features(&hypothesis,
                                           &features[row * ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT],
                                           ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT,
                                           &encoded_count) ||
            encoded_count != ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT) {
            continue;
        }
        int certified = evals[i].train_consistent ? 1 : 0;
        arc_hypothesis_prior_observe(state, &hypothesis, certified);
        targets[row] = certified ? (nn_real)1 : (nn_real)0;
        ++row;
    }

    if (row != 0) {
        nn_real loss = nn_train_batched(state->hypothesis_prior_net,
                                        features,
                                        targets,
                                        row,
                                        row,
                                        learning_rate,
                                        ARC_HYPOTHESIS_PRIOR_EPOCHS,
                                        0);
        if (!isfinite((double)loss)) {
            free(features);
            free(targets);
            return 0;
        }
        state->hypothesis_prior_loss = (double)loss;
        state->hypothesis_prior_samples += row;
        state->hypothesis_prior_updates += ARC_HYPOTHESIS_PRIOR_EPOCHS;
    }

    free(features);
    free(targets);
    return 1;
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
        if (arc_candidate_is_composed(&candidates[i])) {
            continue;
        }
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
            if (arc_candidate_is_composed(&candidates[i])) {
                continue;
            }
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

static int arc_candidate_supports_train_inputs(const ARCTask *task,
                                               const ARCTransformCandidate *candidate)
{
    ARCGrid produced;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output ||
            !arc_apply_candidate(candidate, &task->train[pair].input, &produced)) {
            return 0;
        }
    }
    return task->train_count > 0;
}

static int arc_composed_candidate_exists(const ARCTransformCandidate *candidates,
                                         size_t count,
                                         const ARCTransformCandidate *candidate)
{
    if (!arc_candidate_is_composed(candidate)) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!arc_candidate_is_composed(&candidates[i])) {
            continue;
        }
        if (candidates[i].steps[0].op == candidate->steps[0].op &&
            candidates[i].steps[1].op == candidate->steps[1].op &&
            strcmp(candidates[i].steps[0].name, candidate->steps[0].name) == 0 &&
            strcmp(candidates[i].steps[1].name, candidate->steps[1].name) == 0) {
            return 1;
        }
    }
    return 0;
}

static size_t arc_add_learned_compositions(const ARCTask *task,
                                           ARCProgramLearnedState *learned,
                                           ARCTransformCandidate *candidates,
                                           size_t capacity,
                                           size_t base_count)
{
    if (task == NULL || learned == NULL || learned->trained_tasks == 0 ||
        candidates == NULL || base_count == 0 || base_count >= capacity) {
        return base_count;
    }

    ARCProgramEval evals[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramRank ranks[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    for (size_t i = 0; i < base_count; ++i) {
        NNProgram program;
        memset(&evals[i], 0, sizeof(evals[i]));
        if (arc_candidate_to_program(&candidates[i], &program)) {
            evals[i].program_hash = program.canonical_hash;
        }
    }
    arc_build_program_ranks(task,
                            candidates,
                            evals,
                            base_count,
                            ARC_PROGRAM_STRATEGY_LEARNED,
                            learned,
                            &task_features,
                            ranks);

    size_t top_indices[ARC_LEARNED_COMPOSE_TOP];
    size_t top_count = 0;
    for (size_t rank = 0; rank < base_count && top_count < ARC_LEARNED_COMPOSE_TOP; ++rank) {
        size_t index = ranks[rank].candidate_index;
        if (arc_candidate_is_composed(&candidates[index])) {
            continue;
        }
        top_indices[top_count++] = index;
    }

    size_t count = base_count;
    size_t added = 0;
    for (size_t first = 0; first < top_count; ++first) {
        for (size_t second = 0; second < top_count; ++second) {
            if (count >= capacity || added >= ARC_LEARNED_COMPOSE_LIMIT) {
                return count;
            }
            const ARCTransformCandidate *first_candidate = &candidates[top_indices[first]];
            const ARCTransformCandidate *second_candidate = &candidates[top_indices[second]];
            if (first_candidate->op == ARC_RULE_GRID_IDENTITY &&
                second_candidate->op == ARC_RULE_GRID_IDENTITY) {
                continue;
            }
            ARCTransformCandidate composed;
            if (!arc_init_composed_candidate(&composed, first_candidate, second_candidate) ||
                arc_composed_candidate_exists(candidates, count, &composed) ||
                !arc_candidate_supports_train_inputs(task, &composed)) {
                continue;
            }
            arc_add_candidate(candidates, capacity, &count, &composed);
            ++added;
        }
    }
    return count;
}

static int arc_candidate_name_exists(const ARCTransformCandidate *candidates,
                                     size_t count,
                                     const char *name)
{
    if (candidates == NULL || name == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(candidates[i].name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static int arc_hypothesis_proposal_name_exists(const ARCHypothesisProposal *proposals,
                                               size_t count,
                                               const char *name)
{
    if (proposals == NULL || name == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (strcmp(proposals[i].candidate.name, name) == 0) {
            return 1;
        }
    }
    return 0;
}

static double arc_hypothesis_structural_proposal_score(const ARCProgramLearnedState *learned,
                                                       const ARCTransformCandidate *candidate,
                                                       const ARCObject *input_object,
                                                       const ARCObject *output_object,
                                                       const ARCProgramTaskFeatures *task_features)
{
    (void)task_features;
    if (candidate == NULL || input_object == NULL || output_object == NULL) {
        return -1.0e9;
    }

    double score = 64.0;
    score += arc_objects_same_shape(input_object, output_object) ? 16.0 : -32.0;
    score -= 0.5 * (double)(abs(candidate->dx) + abs(candidate->dy));
    score += candidate->input_color == candidate->output_color ? 0.25 : 1.0;
    switch (arc_candidate_selector_kind(candidate)) {
    case ARC_OBJECT_SELECTOR_UNIQUE_COLOR:
        score += 6.0;
        break;
    case ARC_OBJECT_SELECTOR_LARGEST_COLOR:
    case ARC_OBJECT_SELECTOR_SMALLEST_COLOR:
        score += 4.0;
        break;
    case ARC_OBJECT_SELECTOR_INTERIOR_COLOR:
    case ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR:
        score += 3.0;
        break;
    case ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR:
        score += 2.0;
        break;
    default:
        break;
    }
    if (learned != NULL) {
        score += arc_learned_family_score(learned, candidate);
        score += 8.0 * arc_learned_relation_score(learned, candidate);
    }
    return score;
}

static double arc_hypothesis_slot_proposal_score(ARCProgramLearnedState *learned,
                                                 const NNHypothesis *hypothesis,
                                                 const ARCTransformCandidate *candidate,
                                                 const ARCObject *input_object,
                                                 const ARCObject *output_object,
                                                 const ARCProgramTaskFeatures *task_features)
{
    double structural =
        arc_hypothesis_structural_proposal_score(learned,
                                                 candidate,
                                                 input_object,
                                                 output_object,
                                                 task_features);
    if (learned == NULL || hypothesis == NULL || learned->hypothesis_prior_net == NULL) {
        return structural;
    }

    size_t proposal_support = 0;
    double slot_log_odds =
        arc_hypothesis_slot_log_odds(learned, hypothesis, &proposal_support);
    if (proposal_support < ARC_HYPOTHESIS_PRIOR_MIN_PROPOSALS &&
        learned->hypothesis_prior_samples < ARC_HYPOTHESIS_PRIOR_MIN_PROPOSALS) {
        return structural;
    }

    nn_real features[ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT];
    size_t feature_count = 0;
    nn_real output[1] = {0};
    double neural_score = 0.0;
    if (learned->hypothesis_prior_samples >= ARC_HYPOTHESIS_PRIOR_MIN_PROPOSALS &&
        nn_hypothesis_encode_features(hypothesis,
                                      features,
                                      ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT,
                                      &feature_count) &&
        feature_count == ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT) {
        nn_forward(learned->hypothesis_prior_net, features, output);
        neural_score = (double)output[0];
    }

    double cold_start = 0.0;
    uint8_t selector = arc_candidate_selector_kind(candidate);
    if (selector < ARC_OBJECT_SELECTOR_KIND_COUNT) {
        cold_start += learned->object_selector_score[selector];
        for (size_t relation = 1; relation < ARC_OBJECT_RELATION_KIND_COUNT; ++relation) {
            cold_start += 0.25 * learned->relation_selector_score[relation][selector];
        }
    }

    return structural + 12.0 * slot_log_odds + 24.0 * neural_score + 8.0 * cold_start;
}

static void arc_hypothesis_insert_proposal(ARCHypothesisProposal *proposals,
                                           size_t *proposal_count,
                                           const ARCHypothesisProposal *proposal)
{
    if (proposals == NULL || proposal_count == NULL || proposal == NULL) {
        return;
    }
    size_t count = *proposal_count;
    size_t pos = count;
    if (count < ARC_HYPOTHESIS_PROPOSAL_CAPACITY) {
        proposals[pos] = *proposal;
        ++count;
    } else if (proposal->score > proposals[count - 1u].score) {
        proposals[count - 1u] = *proposal;
        pos = count - 1u;
    } else {
        return;
    }

    while (pos > 0 && proposals[pos].score > proposals[pos - 1u].score) {
        ARCHypothesisProposal tmp = proposals[pos - 1u];
        proposals[pos - 1u] = proposals[pos];
        proposals[pos] = tmp;
        --pos;
    }
    *proposal_count = count;
}

static size_t arc_induce_translate_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                              const ARCTask *task,
                                              ARCProgramLearnedState *learned,
                                              const ARCTransformCandidate *existing_candidates,
                                              size_t existing_count,
                                              ARCHypothesisProposal *proposals,
                                              size_t proposal_capacity,
                                              size_t proposal_count)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_count >= proposal_capacity) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count && proposal_count < proposal_capacity;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0;
             train_pair < task->train_count && proposal_count < proposal_capacity;
             ++train_pair) {
            ARCScene input_scene;
            ARCScene output_scene;
            if (!task->train[train_pair].has_output ||
                !arc_extract_scene(&task->train[train_pair].input,
                                   background,
                                   &input_scene) ||
                !arc_extract_scene(&task->train[train_pair].output,
                                   background,
                                   &output_scene)) {
                continue;
            }
            for (size_t input_index = 0;
                 input_index < input_scene.object_count && proposal_count < proposal_capacity;
                 ++input_index) {
                const ARCObject *input_object = &input_scene.objects[input_index];
                uint8_t selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                uint8_t reference_color = 0;
                if (input_object->color == background ||
                    !arc_infer_selector_for_scene_object(&input_scene,
                                                         input_object,
                                                         &selector_kind,
                                                         &reference_color)) {
                    continue;
                }
                int reference_color_used =
                    selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR;
                for (size_t output_index = 0;
                     output_index < output_scene.object_count && proposal_count < proposal_capacity;
                     ++output_index) {
                    const ARCObject *output_object = &output_scene.objects[output_index];
                    if (output_object->color == background ||
                        !arc_objects_same_shape(input_object, output_object)) {
                        continue;
                    }
                    int dx = (int)output_object->min_x - (int)input_object->min_x;
                    int dy = (int)output_object->min_y - (int)input_object->min_y;
                    if (dx == 0 && dy == 0 &&
                        output_object->color == input_object->color) {
                        continue;
                    }

                    NNHypothesis hypothesis;
                    ARCTransformCandidate candidate;
                    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                        continue;
                    }
                    int ok = nn_hypothesis_bind(&hypothesis,
                                                "object_scene.background",
                                                (double)background,
                                                background);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.selector_kind",
                                                  (double)selector_kind,
                                                  selector_kind);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.color",
                                                  (double)input_object->color,
                                                  input_object->color);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.reference_color_used",
                                                  (double)reference_color_used,
                                                  reference_color_used);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.reference_color",
                                                  (double)reference_color,
                                                  reference_color);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "translate.dx",
                                                  (double)dx,
                                                  dx);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "translate.dy",
                                                  (double)dy,
                                                  dy);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "render.background",
                                                  (double)background,
                                                  background);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "render.output_color",
                                                  (double)output_object->color,
                                                  output_object->color);
                    if (!ok ||
                        !binding->to_candidate(&hypothesis, &candidate) ||
                        arc_candidate_name_exists(existing_candidates,
                                                  existing_count,
                                                  candidate.name) ||
                        arc_hypothesis_proposal_name_exists(proposals,
                                                            proposal_count,
                                                            candidate.name)) {
                        continue;
                    }
                    ARCHypothesisProposal proposal;
                    proposal.hypothesis = hypothesis;
                    proposal.candidate = candidate;
                    proposal.score = arc_hypothesis_slot_proposal_score(learned,
                                                                        &hypothesis,
                                                                        &candidate,
                                                                        input_object,
                                                                        output_object,
                                                                        &task_features);
                    arc_hypothesis_insert_proposal(proposals,
                                                   &proposal_count,
                                                   &proposal);
                }
            }
        }
    }

    return proposal_count;
}

static void arc_consider_move_to_anchor_proposal(
    const ARCHypothesisTemplateBinding *binding,
    ARCProgramLearnedState *learned,
    const ARCTransformCandidate *existing_candidates,
    size_t existing_count,
    ARCHypothesisProposal *proposals,
    size_t *proposal_count,
    size_t proposal_capacity,
    const ARCProgramTaskFeatures *task_features,
    uint8_t background,
    const ARCObject *subject,
    uint8_t subject_selector_kind,
    const ARCObject *anchor,
    uint8_t anchor_selector_kind,
    const ARCObject *output_object,
    int axis,
    int offset)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL || proposals == NULL ||
        proposal_count == NULL || *proposal_count >= proposal_capacity ||
        task_features == NULL ||
        subject == NULL || anchor == NULL || output_object == NULL) {
        return;
    }
    NNHypothesis hypothesis;
    ARCTransformCandidate candidate;
    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
        return;
    }
    int ok = nn_hypothesis_bind(&hypothesis,
                                "object_scene.background",
                                (double)background,
                                background);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "subject.selector_kind",
                                  (double)subject_selector_kind,
                                  subject_selector_kind);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "subject.color",
                                  (double)subject->color,
                                  subject->color);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "anchor.selector_kind",
                                  (double)anchor_selector_kind,
                                  anchor_selector_kind);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "anchor.color",
                                  (double)anchor->color,
                                  anchor->color);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "align.axis",
                                  (double)axis,
                                  axis);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "align.offset",
                                  (double)offset,
                                  offset);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "render.output_color",
                                  (double)output_object->color,
                                  output_object->color);
    if (!ok || !binding->to_candidate(&hypothesis, &candidate) ||
        arc_candidate_name_exists(existing_candidates, existing_count, candidate.name) ||
        arc_hypothesis_proposal_name_exists(proposals,
                                            *proposal_count,
                                            candidate.name)) {
        return;
    }
    ARCHypothesisProposal proposal;
    proposal.hypothesis = hypothesis;
    proposal.candidate = candidate;
    proposal.score = arc_hypothesis_slot_proposal_score(learned,
                                                        &hypothesis,
                                                        &candidate,
                                                        subject,
                                                        output_object,
                                                        task_features);
    arc_hypothesis_insert_proposal(proposals, proposal_count, &proposal);
}

static size_t arc_induce_move_to_anchor_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                   const ARCTask *task,
                                                   ARCProgramLearnedState *learned,
                                                   const ARCTransformCandidate *existing_candidates,
                                                   size_t existing_count,
                                                   ARCHypothesisProposal *proposals,
                                                   size_t proposal_capacity,
                                                   size_t proposal_count)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_count >= proposal_capacity) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count && proposal_count < proposal_capacity;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0;
             train_pair < task->train_count && proposal_count < proposal_capacity;
             ++train_pair) {
            ARCScene input_scene;
            ARCScene output_scene;
            if (!task->train[train_pair].has_output ||
                !arc_extract_scene(&task->train[train_pair].input,
                                   background,
                                   &input_scene) ||
                !arc_extract_scene(&task->train[train_pair].output,
                                   background,
                                   &output_scene)) {
                continue;
            }
            for (size_t subject_index = 0;
                 subject_index < input_scene.object_count && proposal_count < proposal_capacity;
                 ++subject_index) {
                const ARCObject *subject = &input_scene.objects[subject_index];
                uint8_t subject_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                if (subject->color == background ||
                    !arc_infer_non_reference_selector_for_scene_object(
                        &input_scene,
                        subject,
                        &subject_selector_kind)) {
                    continue;
                }
                for (size_t anchor_index = 0;
                     anchor_index < input_scene.object_count && proposal_count < proposal_capacity;
                     ++anchor_index) {
                    const ARCObject *anchor = &input_scene.objects[anchor_index];
                    uint8_t anchor_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                    if (anchor->id == subject->id ||
                        anchor->color == background ||
                        anchor->color == subject->color ||
                        !arc_infer_non_reference_selector_for_scene_object(
                            &input_scene,
                            anchor,
                            &anchor_selector_kind)) {
                        continue;
                    }
                    for (size_t output_index = 0;
                         output_index < output_scene.object_count && proposal_count < proposal_capacity;
                         ++output_index) {
                        const ARCObject *output_object =
                            &output_scene.objects[output_index];
                        if (output_object->color == background ||
                            !arc_objects_same_shape(subject, output_object)) {
                            continue;
                        }
                        int offset_x = (int)output_object->min_x -
                                       (int)anchor->min_x;
                        int offset_y = (int)output_object->min_y -
                                       (int)anchor->min_y;
                        if (output_object->min_y == subject->min_y) {
                            arc_consider_move_to_anchor_proposal(binding,
                                                                 learned,
                                                                 existing_candidates,
                                                                 existing_count,
                                                                 proposals,
                                                                 &proposal_count,
                                                                 proposal_capacity,
                                                                 &task_features,
                                                                 background,
                                                                 subject,
                                                                 subject_selector_kind,
                                                                 anchor,
                                                                 anchor_selector_kind,
                                                                 output_object,
                                                                 0,
                                                                 offset_x);
                        }
                        if (output_object->min_x == subject->min_x) {
                            arc_consider_move_to_anchor_proposal(binding,
                                                                 learned,
                                                                 existing_candidates,
                                                                 existing_count,
                                                                 proposals,
                                                                 &proposal_count,
                                                                 proposal_capacity,
                                                                 &task_features,
                                                                 background,
                                                                 subject,
                                                                 subject_selector_kind,
                                                                 anchor,
                                                                 anchor_selector_kind,
                                                                 output_object,
                                                                 1,
                                                                 offset_y);
                        }
                        if (offset_x == offset_y) {
                            arc_consider_move_to_anchor_proposal(binding,
                                                                 learned,
                                                                 existing_candidates,
                                                                 existing_count,
                                                                 proposals,
                                                                 &proposal_count,
                                                                 proposal_capacity,
                                                                 &task_features,
                                                                 background,
                                                                 subject,
                                                                 subject_selector_kind,
                                                                 anchor,
                                                                 anchor_selector_kind,
                                                                 output_object,
                                                                 2,
                                                                 offset_x);
                        }
                    }
                }
            }
        }
    }

    return proposal_count;
}

static void arc_hypothesis_set_test_coverage(const ARCTask *task,
                                             const ARCTransformCandidate *candidate,
                                             NNProgramCoverage *coverage,
                                             size_t program_index)
{
    ARCGrid predictions[ARC_MAX_PAIRS];
    size_t unsupported = 0;
    if (task == NULL || candidate == NULL || coverage == NULL ||
        !arc_apply_candidate_to_tests(task, candidate, predictions, &unsupported)) {
        return;
    }

    size_t event = 0;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        if (!task->test[pair].has_output) {
            continue;
        }
        size_t cells = arc_grid_cells(&task->test[pair].output);
        if (!arc_grid_same_shape(&predictions[pair], &task->test[pair].output)) {
            event += cells;
            continue;
        }
        for (size_t cell = 0; cell < cells; ++cell) {
            if (predictions[pair].cells[cell] == task->test[pair].output.cells[cell]) {
                nn_program_coverage_set(coverage, program_index, event);
            }
            ++event;
        }
    }
}

static size_t arc_select_hypothesis_proposals(const ARCTask *task,
                                              const ARCHypothesisProposal *proposals,
                                              size_t proposal_count,
                                              ARCTransformCandidate *candidates,
                                              size_t capacity,
                                              size_t base_count)
{
    if (task == NULL || proposals == NULL || candidates == NULL ||
        base_count >= capacity || proposal_count == 0) {
        return base_count;
    }

    ARCHypothesisProposal valid[ARC_HYPOTHESIS_PROPOSAL_CAPACITY];
    size_t valid_count = 0;
    for (size_t i = 0; i < proposal_count && valid_count < ARC_HYPOTHESIS_PROPOSAL_CAPACITY; ++i) {
        size_t checked_cells = 0;
        size_t mismatches = 0;
        size_t unsupported = 0;
        if (arc_candidate_name_exists(candidates, base_count, proposals[i].candidate.name) ||
            !arc_candidate_matches_train(task,
                                         &proposals[i].candidate,
                                         &checked_cells,
                                         &mismatches,
                                         &unsupported)) {
            continue;
        }
        valid[valid_count] = proposals[i];
        valid[valid_count].hypothesis.evidence_support = checked_cells;
        valid[valid_count].hypothesis.evidence_residual_gain =
            checked_cells == 0 ? 0.0 : 1.0 - (double)mismatches / (double)checked_cells;
        ++valid_count;
    }
    if (valid_count == 0) {
        return base_count;
    }

    size_t count = base_count;
    size_t task_cells = arc_task_test_cells(task);
    NNProgramCoverage coverage;
    memset(&coverage, 0, sizeof(coverage));
    uint64_t *covered = NULL;
    int have_coverage = task_cells > 0 &&
                        nn_program_coverage_init(&coverage, valid_count, task_cells);
    if (have_coverage && coverage.word_count != 0) {
        covered = calloc(coverage.word_count, sizeof(*covered));
        if (covered == NULL) {
            nn_program_coverage_free(&coverage);
            have_coverage = 0;
        }
    }
    if (have_coverage) {
        for (size_t i = 0; i < valid_count; ++i) {
            arc_hypothesis_set_test_coverage(task,
                                             &valid[i].candidate,
                                             &coverage,
                                             i);
        }
    }

    uint8_t selected[ARC_HYPOTHESIS_PROPOSAL_CAPACITY];
    memset(selected, 0, sizeof(selected));
    size_t selected_count = 0;
    if (have_coverage && covered != NULL) {
        for (size_t step = 0;
             step < ARC_HYPOTHESIS_TOP_K && count < capacity;
             ++step) {
            size_t best = (size_t)-1;
            size_t best_gain = 0;
            double best_score = -1.0e300;
            for (size_t i = 0; i < valid_count; ++i) {
                if (selected[i]) {
                    continue;
                }
                size_t gain = nn_program_coverage_gain(&coverage, i, covered);
                if (best == (size_t)-1 ||
                    gain > best_gain ||
                    (gain == best_gain && valid[i].score > best_score)) {
                    best = i;
                    best_gain = gain;
                    best_score = valid[i].score;
                }
            }
            if (best == (size_t)-1 || best_gain == 0) {
                break;
            }
            selected[best] = 1;
            ++selected_count;
            nn_program_coverage_add_program_to_residual(&coverage, best, covered);
            if (!arc_candidate_name_exists(candidates, count, valid[best].candidate.name)) {
                arc_add_candidate(candidates, capacity, &count, &valid[best].candidate);
            }
        }
    }

    if (selected_count == 0) {
        size_t proposal_limit = valid_count < ARC_HYPOTHESIS_TOP_K
                                    ? valid_count
                                    : ARC_HYPOTHESIS_TOP_K;
        for (size_t i = 0; i < proposal_limit && count < capacity; ++i) {
            if (!arc_candidate_name_exists(candidates, count, valid[i].candidate.name)) {
                arc_add_candidate(candidates, capacity, &count, &valid[i].candidate);
            }
        }
    }

    free(covered);
    if (have_coverage) {
        nn_program_coverage_free(&coverage);
    }
    return count;
}

static size_t arc_build_candidate_set(const ARCTask *task,
                                      ARCProgramLearnedState *learned,
                                      ARCTransformCandidate *candidates,
                                      size_t capacity,
                                      int include_hypothesis_induction)
{
    size_t count = arc_build_candidates(task, candidates, capacity);
    if (include_hypothesis_induction) {
        ARCHypothesisProposal proposals[ARC_HYPOTHESIS_PROPOSAL_CAPACITY];
        size_t proposal_count = 0;
        size_t binding_count = 0;
        const ARCHypothesisTemplateBinding *bindings =
            arc_hypothesis_template_registry(&binding_count);
        for (size_t i = 0;
             i < binding_count && proposal_count < ARC_HYPOTHESIS_PROPOSAL_CAPACITY;
             ++i) {
            if (bindings[i].induce == NULL) {
                continue;
            }
            proposal_count = bindings[i].induce(&bindings[i],
                                                task,
                                                learned,
                                                candidates,
                                                count,
                                                proposals,
                                                ARC_HYPOTHESIS_PROPOSAL_CAPACITY,
                                                proposal_count);
        }
        count = arc_select_hypothesis_proposals(task,
                                                proposals,
                                                proposal_count,
                                                candidates,
                                                capacity,
                                                count);
    }
    return arc_add_learned_compositions(task, learned, candidates, capacity, count);
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

static int arc_program_learned_observe(ARCProgramLearnedState *state,
                                       const ARCTask *task,
                                       const ARCTransformCandidate *candidates,
                                       const ARCProgramEval *evals,
                                       size_t candidate_count,
                                       nn_real learning_rate)
{
    size_t family_seen[32];
    size_t family_success[32];
    size_t relation_seen[ARC_OBJECT_RELATION_KIND_COUNT];
    size_t relation_success[ARC_OBJECT_RELATION_KIND_COUNT];
    size_t relation_color_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t relation_color_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t relation_background_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT];
    size_t relation_background_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT];
    size_t relation_selector_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_selector_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_output_selector_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_output_selector_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_render_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_RENDER_KIND_COUNT];
    size_t relation_render_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_RENDER_KIND_COUNT];
    size_t object_selector_seen[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t object_selector_success[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t object_color_seen[ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t object_color_success[ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t object_render_seen[ARC_OBJECT_RENDER_KIND_COUNT];
    size_t object_render_success[ARC_OBJECT_RENDER_KIND_COUNT];
    memset(family_seen, 0, sizeof(family_seen));
    memset(family_success, 0, sizeof(family_success));
    memset(relation_seen, 0, sizeof(relation_seen));
    memset(relation_success, 0, sizeof(relation_success));
    memset(relation_color_seen, 0, sizeof(relation_color_seen));
    memset(relation_color_success, 0, sizeof(relation_color_success));
    memset(relation_background_seen, 0, sizeof(relation_background_seen));
    memset(relation_background_success, 0, sizeof(relation_background_success));
    memset(relation_selector_seen, 0, sizeof(relation_selector_seen));
    memset(relation_selector_success, 0, sizeof(relation_selector_success));
    memset(relation_output_selector_seen, 0, sizeof(relation_output_selector_seen));
    memset(relation_output_selector_success, 0, sizeof(relation_output_selector_success));
    memset(relation_render_seen, 0, sizeof(relation_render_seen));
    memset(relation_render_success, 0, sizeof(relation_render_success));
    memset(object_selector_seen, 0, sizeof(object_selector_seen));
    memset(object_selector_success, 0, sizeof(object_selector_success));
    memset(object_color_seen, 0, sizeof(object_color_seen));
    memset(object_color_success, 0, sizeof(object_color_success));
    memset(object_render_seen, 0, sizeof(object_render_seen));
    memset(object_render_success, 0, sizeof(object_render_success));
    for (size_t i = 0; i < candidate_count; ++i) {
        if (arc_candidate_is_composed(&candidates[i])) {
            continue;
        }
        size_t family = arc_candidate_family(&candidates[i]);
        if (family >= sizeof(family_seen) / sizeof(family_seen[0])) {
            continue;
        }
        ++family_seen[family];
        if (evals[i].train_consistent) {
            ++family_success[family];
        }
        size_t relation_kind = arc_object_relation_kind(&candidates[i]);
        if (relation_kind != 0 &&
            relation_kind < ARC_OBJECT_RELATION_KIND_COUNT &&
            candidates[i].input_color < ARC_COLOR_COUNT &&
            candidates[i].output_color < ARC_COLOR_COUNT) {
            uint8_t input_color = candidates[i].input_color;
            uint8_t output_color = candidates[i].output_color;
            uint8_t background = candidates[i].background_color;
            uint8_t selector = arc_candidate_selector_kind(&candidates[i]);
            uint8_t output_selector = arc_candidate_output_selector_kind(&candidates[i]);
            uint8_t render_mode = arc_candidate_render_mode(&candidates[i]);
            ++relation_seen[relation_kind];
            ++relation_color_seen[relation_kind][input_color][output_color];
            ++relation_background_seen[relation_kind][background];
            ++relation_selector_seen[relation_kind][selector];
            ++relation_output_selector_seen[relation_kind][output_selector];
            ++relation_render_seen[relation_kind][render_mode];
            if (evals[i].train_consistent) {
                ++relation_success[relation_kind];
                ++relation_color_success[relation_kind][input_color][output_color];
                ++relation_background_success[relation_kind][background];
                ++relation_selector_success[relation_kind][selector];
                ++relation_output_selector_success[relation_kind][output_selector];
                ++relation_render_success[relation_kind][render_mode];
            }
        }
        if (arc_candidate_is_selected_object_program(&candidates[i]) &&
            candidates[i].input_color < ARC_COLOR_COUNT &&
            candidates[i].output_color < ARC_COLOR_COUNT) {
            uint8_t selector = arc_candidate_selector_kind(&candidates[i]);
            uint8_t render_mode = arc_candidate_render_mode(&candidates[i]);
            ++object_selector_seen[selector];
            ++object_color_seen[candidates[i].input_color][candidates[i].output_color];
            ++object_render_seen[render_mode];
            if (evals[i].train_consistent) {
                ++object_selector_success[selector];
                ++object_color_success[candidates[i].input_color][candidates[i].output_color];
                ++object_render_success[render_mode];
            }
        }
    }
    for (size_t family = 0; family < sizeof(family_seen) / sizeof(family_seen[0]); ++family) {
        if (family_seen[family] == 0) {
            continue;
        }
        double observation = (double)family_success[family] / (double)family_seen[family];
        state->family_score[family] = 0.85 * state->family_score[family] + 0.15 * observation;
    }
    for (size_t relation_kind = 1; relation_kind < ARC_OBJECT_RELATION_KIND_COUNT; ++relation_kind) {
        if (relation_seen[relation_kind] != 0) {
            double observation =
                (double)relation_success[relation_kind] / (double)relation_seen[relation_kind];
            state->relation_score[relation_kind] =
                0.85 * state->relation_score[relation_kind] + 0.15 * observation;
        }
        for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
            for (uint8_t output_color = 0; output_color < ARC_COLOR_COUNT; ++output_color) {
                if (relation_color_seen[relation_kind][input_color][output_color] == 0) {
                    continue;
                }
                double observation =
                    (double)relation_color_success[relation_kind][input_color][output_color] /
                    (double)relation_color_seen[relation_kind][input_color][output_color];
                state->relation_color_score[relation_kind][input_color][output_color] =
                    0.85 * state->relation_color_score[relation_kind][input_color][output_color] +
                    0.15 * observation;
            }
        }
        for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
            if (relation_background_seen[relation_kind][background] == 0) {
                continue;
            }
            double observation =
                (double)relation_background_success[relation_kind][background] /
                (double)relation_background_seen[relation_kind][background];
            state->relation_background_score[relation_kind][background] =
                0.85 * state->relation_background_score[relation_kind][background] +
                0.15 * observation;
        }
        for (uint8_t selector = 0; selector < ARC_OBJECT_SELECTOR_KIND_COUNT; ++selector) {
            if (relation_selector_seen[relation_kind][selector] == 0) {
                continue;
            }
            double observation =
                (double)relation_selector_success[relation_kind][selector] /
                (double)relation_selector_seen[relation_kind][selector];
            state->relation_selector_score[relation_kind][selector] =
                0.85 * state->relation_selector_score[relation_kind][selector] +
                0.15 * observation;
        }
        for (uint8_t selector = 0; selector < ARC_OBJECT_SELECTOR_KIND_COUNT; ++selector) {
            if (relation_output_selector_seen[relation_kind][selector] == 0) {
                continue;
            }
            double observation =
                (double)relation_output_selector_success[relation_kind][selector] /
                (double)relation_output_selector_seen[relation_kind][selector];
            state->relation_output_selector_score[relation_kind][selector] =
                0.85 * state->relation_output_selector_score[relation_kind][selector] +
                0.15 * observation;
        }
        for (uint8_t render_mode = 0; render_mode < ARC_OBJECT_RENDER_KIND_COUNT; ++render_mode) {
            if (relation_render_seen[relation_kind][render_mode] == 0) {
                continue;
            }
            double observation =
                (double)relation_render_success[relation_kind][render_mode] /
                (double)relation_render_seen[relation_kind][render_mode];
            state->relation_render_score[relation_kind][render_mode] =
                0.85 * state->relation_render_score[relation_kind][render_mode] +
                0.15 * observation;
        }
    }
    for (uint8_t selector = 0; selector < ARC_OBJECT_SELECTOR_KIND_COUNT; ++selector) {
        if (object_selector_seen[selector] == 0) {
            continue;
        }
        double observation =
            (double)object_selector_success[selector] / (double)object_selector_seen[selector];
        state->object_selector_score[selector] =
            0.85 * state->object_selector_score[selector] + 0.15 * observation;
    }
    for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
        for (uint8_t output_color = 0; output_color < ARC_COLOR_COUNT; ++output_color) {
            if (object_color_seen[input_color][output_color] == 0) {
                continue;
            }
            double observation =
                (double)object_color_success[input_color][output_color] /
                (double)object_color_seen[input_color][output_color];
            state->object_color_score[input_color][output_color] =
                0.85 * state->object_color_score[input_color][output_color] +
                0.15 * observation;
        }
    }
    for (uint8_t render_mode = 0; render_mode < ARC_OBJECT_RENDER_KIND_COUNT; ++render_mode) {
        if (object_render_seen[render_mode] == 0) {
            continue;
        }
        double observation =
            (double)object_render_success[render_mode] / (double)object_render_seen[render_mode];
        state->object_render_score[render_mode] =
            0.85 * state->object_render_score[render_mode] + 0.15 * observation;
    }
    return arc_program_learned_train_hypothesis_prior(state,
                                                      candidates,
                                                      evals,
                                                      candidate_count,
                                                      learning_rate) &&
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

static const NNHypothesisTemplate *arc_hypothesis_template_object_translate(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "select_then_translate",
                                1024,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     110);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     111);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "translate",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     112);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_GRID,
                                     113);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.reference_color_used",
                                    NN_HYPOTHESIS_SLOT_BOOLEAN,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.reference_color",
                                    NN_HYPOTHESIS_SLOT_REFERENCE_COLOR,
                                    0, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "translate.dx",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "translate.dy",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t selector_kind;
    uint8_t input_color;
    uint8_t reference_color_used;
    uint8_t reference_color;
    int dx;
    int dy;
    uint8_t output_color;
} ARCTranslateHypothesisSpec;

static int arc_translate_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                              ARCTranslateHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_object_translate() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t selector_kind = 0;
    int64_t input_color = 0;
    int64_t reference_color_used = 0;
    int64_t reference_color = 0;
    int64_t dx = 0;
    int64_t dy = 0;
    int64_t render_background = 0;
    int64_t output_color = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis,
                           "select.selector_kind",
                           NULL,
                           &selector_kind) ||
        !nn_hypothesis_get(hypothesis, "select.color", NULL, &input_color) ||
        !nn_hypothesis_get(hypothesis,
                           "select.reference_color_used",
                           NULL,
                           &reference_color_used) ||
        !nn_hypothesis_get(hypothesis,
                           "select.reference_color",
                           NULL,
                           &reference_color) ||
        !nn_hypothesis_get(hypothesis, "translate.dx", NULL, &dx) ||
        !nn_hypothesis_get(hypothesis, "translate.dy", NULL, &dy) ||
        !nn_hypothesis_get(hypothesis,
                           "render.background",
                           NULL,
                           &render_background) ||
        !nn_hypothesis_get(hypothesis,
                           "render.output_color",
                           NULL,
                           &output_color)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        render_background != background ||
        selector_kind < 0 || selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        input_color < 0 || input_color >= ARC_COLOR_COUNT ||
        reference_color_used < 0 || reference_color_used > 1 ||
        reference_color < 0 || reference_color >= ARC_COLOR_COUNT ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT ||
        dx < -ARC_MAX_GRID_WIDTH || dx > ARC_MAX_GRID_WIDTH ||
        dy < -ARC_MAX_GRID_HEIGHT || dy > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    int reference_used = reference_color_used != 0;
    if ((reference_used &&
         selector_kind != ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (!reference_used &&
         selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (!reference_used && reference_color != 0)) {
        return 0;
    }

    spec->background = (uint8_t)background;
    spec->selector_kind = (uint8_t)selector_kind;
    spec->input_color = (uint8_t)input_color;
    spec->reference_color_used = (uint8_t)reference_color_used;
    spec->reference_color = (uint8_t)reference_color;
    spec->dx = (int)dx;
    spec->dy = (int)dy;
    spec->output_color = (uint8_t)output_color;
    return 1;
}

static int arc_candidate_from_translate_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCTransformCandidate *candidate)
{
    ARCTranslateHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_translate_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }

    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = spec.selector_kind;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->input_color = spec.input_color;
    candidate->reference_color = spec.reference_color;
    candidate->output_color = spec.output_color;
    candidate->dx = spec.dx;
    candidate->dy = spec.dy;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    if (spec.reference_color_used) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "hypothesis_%s_%s_ref_%u_bg_%u_color_%u_to_%u_%+d_%+d",
                 hypothesis->template_ref->name,
                 arc_object_selector_short_name(candidate->selector_kind),
                 (unsigned)candidate->reference_color,
                 (unsigned)candidate->background_color,
                 (unsigned)candidate->input_color,
                 (unsigned)candidate->output_color,
                 candidate->dx,
                 candidate->dy);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "hypothesis_%s_%s_ref_none_bg_%u_color_%u_to_%u_%+d_%+d",
                 hypothesis->template_ref->name,
                 arc_object_selector_short_name(candidate->selector_kind),
                 (unsigned)candidate->background_color,
                 (unsigned)candidate->input_color,
                 (unsigned)candidate->output_color,
                 candidate->dx,
                 candidate->dy);
    }
    return 1;
}

static int arc_apply_translate_hypothesis(const NNHypothesis *hypothesis,
                                          const ARCGrid *input,
                                          ARCGrid *output)
{
    ARCTranslateHypothesisSpec spec;
    ARCScene scene;
    ARCObject object;
    if (!arc_translate_spec_from_hypothesis(hypothesis, &spec) ||
        !arc_extract_scene(input, spec.background, &scene) ||
        !arc_select_scene_object(&scene,
                                 spec.selector_kind,
                                 spec.input_color,
                                 spec.reference_color,
                                 &object)) {
        return 0;
    }

    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec.background;
    }
    for (size_t y = object.min_y; y <= object.max_y; ++y) {
        for (size_t x = object.min_x; x <= object.max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != spec.input_color) {
                continue;
            }
            int out_x = (int)x + spec.dx;
            int out_y = (int)y + spec.dy;
            if (out_x < 0 || out_y < 0 ||
                out_x >= (int)input->width || out_y >= (int)input->height) {
                return 0;
            }
            output->cells[(size_t)out_y * output->width + (size_t)out_x] =
                spec.output_color;
        }
    }
    return 1;
}

static const NNHypothesisTemplate *arc_hypothesis_template_move_to_anchor(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "move_to_anchor",
                                1025,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     120);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select_subject",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     121);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select_anchor",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     122);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "align",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     123);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_GRID,
                                     124);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "subject.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "subject.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "anchor.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "anchor.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "align.axis",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "align.offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t subject_selector_kind;
    uint8_t subject_color;
    uint8_t anchor_selector_kind;
    uint8_t anchor_color;
    int axis;
    int offset;
    uint8_t output_color;
} ARCMoveToAnchorHypothesisSpec;

static const char *arc_anchor_axis_short_name(int axis)
{
    switch (axis) {
    case 0:
        return "x";
    case 1:
        return "y";
    case 2:
        return "both";
    }
    return "unknown";
}

static int arc_move_to_anchor_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCMoveToAnchorHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_move_to_anchor() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t subject_selector_kind = 0;
    int64_t subject_color = 0;
    int64_t anchor_selector_kind = 0;
    int64_t anchor_color = 0;
    int64_t axis = 0;
    int64_t offset = 0;
    int64_t output_color = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis,
                           "subject.selector_kind",
                           NULL,
                           &subject_selector_kind) ||
        !nn_hypothesis_get(hypothesis,
                           "subject.color",
                           NULL,
                           &subject_color) ||
        !nn_hypothesis_get(hypothesis,
                           "anchor.selector_kind",
                           NULL,
                           &anchor_selector_kind) ||
        !nn_hypothesis_get(hypothesis,
                           "anchor.color",
                           NULL,
                           &anchor_color) ||
        !nn_hypothesis_get(hypothesis, "align.axis", NULL, &axis) ||
        !nn_hypothesis_get(hypothesis, "align.offset", NULL, &offset) ||
        !nn_hypothesis_get(hypothesis,
                           "render.output_color",
                           NULL,
                           &output_color)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        subject_selector_kind <= ARC_OBJECT_SELECTOR_ALL_COLOR ||
        subject_selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        anchor_selector_kind <= ARC_OBJECT_SELECTOR_ALL_COLOR ||
        anchor_selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        subject_selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR ||
        anchor_selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR ||
        subject_color < 0 || subject_color >= ARC_COLOR_COUNT ||
        anchor_color < 0 || anchor_color >= ARC_COLOR_COUNT ||
        subject_color == anchor_color ||
        axis < 0 || axis > 2 ||
        offset < -ARC_MAX_GRID_WIDTH || offset > ARC_MAX_GRID_WIDTH ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->subject_selector_kind = (uint8_t)subject_selector_kind;
    spec->subject_color = (uint8_t)subject_color;
    spec->anchor_selector_kind = (uint8_t)anchor_selector_kind;
    spec->anchor_color = (uint8_t)anchor_color;
    spec->axis = (int)axis;
    spec->offset = (int)offset;
    spec->output_color = (uint8_t)output_color;
    return 1;
}

static int arc_candidate_from_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                                       ARCTransformCandidate *candidate)
{
    ARCMoveToAnchorHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_move_to_anchor_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = spec.subject_selector_kind;
    candidate->output_selector_kind = spec.anchor_selector_kind;
    candidate->input_color = spec.subject_color;
    candidate->reference_color = spec.anchor_color;
    candidate->output_color = spec.output_color;
    candidate->dx = spec.axis;
    candidate->dy = spec.offset;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_subject_%s_%u_anchor_%s_%u_axis_%s_off_%+d_bg_%u_to_%u",
             hypothesis->template_ref->name,
             arc_object_selector_short_name(spec.subject_selector_kind),
             (unsigned)spec.subject_color,
             arc_object_selector_short_name(spec.anchor_selector_kind),
             (unsigned)spec.anchor_color,
             arc_anchor_axis_short_name(spec.axis),
             spec.offset,
             (unsigned)spec.background,
             (unsigned)spec.output_color);
    return 1;
}

static int arc_apply_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                               const ARCGrid *input,
                                               ARCGrid *output)
{
    ARCMoveToAnchorHypothesisSpec spec;
    ARCScene scene;
    ARCObject subject;
    ARCObject anchor;
    if (!arc_move_to_anchor_spec_from_hypothesis(hypothesis, &spec) ||
        !arc_extract_scene(input, spec.background, &scene) ||
        !arc_select_scene_object(&scene,
                                 spec.subject_selector_kind,
                                 spec.subject_color,
                                 0,
                                 &subject) ||
        !arc_select_scene_object(&scene,
                                 spec.anchor_selector_kind,
                                 spec.anchor_color,
                                 0,
                                 &anchor)) {
        return 0;
    }

    int out_min_x = (int)subject.min_x;
    int out_min_y = (int)subject.min_y;
    if (spec.axis == 0 || spec.axis == 2) {
        out_min_x = (int)anchor.min_x + spec.offset;
    }
    if (spec.axis == 1 || spec.axis == 2) {
        out_min_y = (int)anchor.min_y + spec.offset;
    }

    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec.background;
    }
    for (size_t y = subject.min_y; y <= subject.max_y; ++y) {
        for (size_t x = subject.min_x; x <= subject.max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != spec.subject_color) {
                continue;
            }
            int out_x = out_min_x + (int)x - (int)subject.min_x;
            int out_y = out_min_y + (int)y - (int)subject.min_y;
            if (out_x < 0 || out_y < 0 ||
                out_x >= (int)input->width || out_y >= (int)input->height) {
                return 0;
            }
            output->cells[(size_t)out_y * output->width + (size_t)out_x] =
                spec.output_color;
        }
    }
    return 1;
}

static const ARCHypothesisTemplateBinding *arc_hypothesis_template_registry(size_t *count)
{
    static const ARCHypothesisTemplateBinding bindings[] = {
        {
            "select_then_translate",
            1024,
            arc_hypothesis_template_object_translate,
            arc_induce_translate_hypotheses,
            arc_candidate_from_translate_hypothesis,
            arc_apply_translate_hypothesis,
        },
        {
            "move_to_anchor",
            1025,
            arc_hypothesis_template_move_to_anchor,
            arc_induce_move_to_anchor_hypotheses,
            arc_candidate_from_move_to_anchor_hypothesis,
            arc_apply_move_to_anchor_hypothesis,
        },
    };
    if (count != NULL) {
        *count = sizeof(bindings) / sizeof(bindings[0]);
    }
    return bindings;
}

static const ARCHypothesisTemplateBinding *arc_hypothesis_binding_for_template_id(uint32_t template_id)
{
    size_t count = 0;
    const ARCHypothesisTemplateBinding *bindings =
        arc_hypothesis_template_registry(&count);
    for (size_t i = 0; i < count; ++i) {
        if (bindings[i].template_id == template_id) {
            return &bindings[i];
        }
    }
    return NULL;
}

static int arc_apply_hypothesis_candidate(const ARCTransformCandidate *candidate,
                                          const ARCGrid *input,
                                          ARCGrid *output)
{
    NNHypothesis hypothesis;
    if (!arc_hypothesis_from_candidate(candidate, &hypothesis)) {
        return 0;
    }
    const ARCHypothesisTemplateBinding *binding =
        arc_hypothesis_binding_for_template_id(candidate->hypothesis_template_id);
    if (binding == NULL || binding->apply == NULL) {
        return 0;
    }
    return binding->apply(&hypothesis, input, output);
}

static void arc_emit_hypothesis_evidence_json(const char *task_id,
                                              const NNHypothesis *hypothesis,
                                              const NNProgram *program,
                                              const ARCProgramEval *eval,
                                              size_t test_cells)
{
    if (hypothesis == NULL || hypothesis->template_ref == NULL ||
        program == NULL) {
        return;
    }
    size_t events = eval != NULL && eval->test_cells != 0
                        ? eval->test_cells
                        : test_cells;
    size_t covered = eval != NULL ? eval->test_correct_cells : 0;
    NNHypothesisEvidence evidence = {
        .task = "arc_agi",
        .task_instance = task_id != NULL ? task_id : "unknown",
        .program = program,
        .certificate_status = eval != NULL && eval->train_consistent && eval->test_supported
                                  ? NN_CERTIFICATE_CERTIFIED
                                  : NN_CERTIFICATE_REJECTED,
        .proof_kind = eval != NULL && eval->train_consistent
                          ? "arc_train_pairs_consistent"
                          : "arc_hypothesis_unchecked",
        .reason = eval != NULL && eval->train_consistent
                      ? "hypothesis-derived transform matches train pairs"
                      : "hypothesis-derived transform did not certify on train pairs",
        .event_count = events,
        .covered_events = covered,
        .support = eval != NULL ? eval->train_checked_cells : 0,
        .false_rejections = eval != NULL ? eval->train_mismatches : 0,
        .utility = events == 0 ? 0.0 : (double)covered / (double)events,
    };
    nn_hypothesis_write_evidence_json(stdout, hypothesis, &evidence);
}

static size_t arc_emit_induction_smoke(const ARCTask *task,
                                       const ARCTransformCandidate *candidates,
                                       const ARCProgramEval *evals,
                                       size_t candidate_count)
{
    size_t test_cells = arc_task_test_cells(task);
    size_t emitted = 0;
    for (size_t i = 0; i < candidate_count && emitted < 8; ++i) {
        if (!arc_candidate_is_hypothesis_induced(&candidates[i])) {
            continue;
        }
        NNHypothesis hypothesis;
        NNProgram program;
        if (!arc_hypothesis_from_candidate(&candidates[i], &hypothesis) ||
            !nn_hypothesis_compile(&hypothesis, &program)) {
            continue;
        }
        arc_emit_hypothesis_evidence_json(task->id,
                                          &hypothesis,
                                          &program,
                                          &evals[i],
                                          test_cells);
        ++emitted;
    }
    return emitted;
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
    size_t candidate_count = arc_build_candidate_set(task,
                                                     &learned,
                                                     candidates,
                                                     ARC_RULE_CANDIDATE_COUNT,
                                                     config != NULL && config->induction_smoke);
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
    if (config != NULL && config->induction_smoke) {
        arc_emit_induction_smoke(task, candidates, evals, candidate_count);
    }
    arc_program_learned_free(&learned);
}

static ARCTransformMetrics evaluate_arc_transform_gate_candidates(const ARCTask *task,
                                                                 const ARCTransformCandidate *candidates,
                                                                 size_t candidate_count)
{
    ARCTransformMetrics metrics;
    ARCGrid first_predictions[ARC_MAX_PAIRS];
    ARCGrid current_predictions[ARC_MAX_PAIRS];
    memset(&metrics, 0, sizeof(metrics));
    memset(first_predictions, 0, sizeof(first_predictions));
    metrics.unique_test_prediction = 1;
    metrics.test_pairs = task->test_count;
    metrics.rules = candidate_count;

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

static ARCTransformMetrics evaluate_arc_transform_gate_task(const ARCTask *task)
{
    ARCTransformCandidate candidates[ARC_RULE_CANDIDATE_COUNT];
    size_t candidate_count = arc_build_candidates(task, candidates, ARC_RULE_CANDIDATE_COUNT);
    return evaluate_arc_transform_gate_candidates(task, candidates, candidate_count);
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
                                         nn_real learning_rate,
                                         int include_hypothesis_induction)
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
        size_t candidate_count = include_hypothesis_induction
                                     ? arc_build_candidate_set(&task,
                                                               learned_state,
                                                               candidates,
                                                               ARC_RULE_CANDIDATE_COUNT,
                                                               1)
                                     : arc_build_candidates(&task,
                                                            candidates,
                                                            ARC_RULE_CANDIDATE_COUNT);
        arc_evaluate_program_candidates(&task, candidates, candidate_count, evals);
        if (!arc_program_learned_observe(learned_state,
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
                                       config->program_learning_rate,
                                       config->induction_smoke)) {
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

    size_t induction_evidence_rows = 0;
    for (size_t i = 0; i < file_count; ++i) {
        if (config->max_tasks != 0 && summary.loaded_tasks >= config->max_tasks) {
            break;
        }
        ARCTask task;
        if (!arc_load_task_file(files[i].path, files[i].id, &task)) {
            ++summary.parse_errors;
            continue;
        }
        ARCTransformCandidate candidates[ARC_RULE_CANDIDATE_COUNT];
        ARCProgramEval evals[ARC_RULE_CANDIDATE_COUNT];
        size_t candidate_count = arc_build_candidate_set(&task,
                                                         &learned_state,
                                                         candidates,
                                                         ARC_RULE_CANDIDATE_COUNT,
                                                         config->induction_smoke);
        ARCTransformMetrics transform =
            evaluate_arc_transform_gate_candidates(&task, candidates, candidate_count);
        arc_loader_summary_add(&summary, &task, &transform);
        arc_evaluate_program_candidates(&task, candidates, candidate_count, evals);
        if (config->induction_smoke) {
            induction_evidence_rows +=
                arc_emit_induction_smoke(&task, candidates, evals, candidate_count);
        }
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
            ARCTransformCandidate learn_candidates[ARC_RULE_CANDIDATE_COUNT];
            ARCProgramEval learn_evals[ARC_RULE_CANDIDATE_COUNT];
            size_t learn_candidate_count = arc_build_candidates(&task,
                                                                learn_candidates,
                                                                ARC_RULE_CANDIDATE_COUNT);
            arc_evaluate_program_candidates(&task,
                                            learn_candidates,
                                            learn_candidate_count,
                                            learn_evals);
            if (!arc_program_learned_observe(&learned_state,
                                             &task,
                                             learn_candidates,
                                             learn_evals,
                                             learn_candidate_count,
                                             config->program_learning_rate)) {
                arc_program_learned_free(&learned_state);
                free(files);
                return 0;
            }
        }
        if (!config->loader_summary_only) {
            print_arc_transform_json(task.id, &transform);
            if (transform.certified) {
                ARCMetrics transform_metrics = arc_transform_to_metrics(&task, &transform);
                print_arc_metrics_json(task.id,
                                       "symbolic-transform-v10",
                                       "certified_transform_gate",
                                       "arc-transform-gate-v10",
                                       &transform_metrics);
            }
        }
    }
    if (config->induction_smoke &&
        induction_evidence_rows < config->min_induction_evidence) {
        fprintf(stderr,
                "ARC induction emitted %zu hypothesis evidence rows; expected at least %zu\n",
                induction_evidence_rows,
                config->min_induction_evidence);
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
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
          "[--loader-summary-only] [--induction] [--min-induction-evidence count]\n",
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
    config->min_induction_evidence = 1;
    config->seed = 20240516u;
    config->learning_rate = (nn_real)0.006;
    config->program_learning_rate = (nn_real)0.015;
    config->save_path = NULL;
    config->task_dir = NULL;
    config->program_train_dir = NULL;
    config->loader_summary_only = 0;
    config->program_budget_grid = 0;
    config->induction_smoke = 0;
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
        } else if (strcmp(argv[i], "--induction") == 0) {
            config->induction_smoke = 1;
        } else if (strcmp(argv[i], "--min-induction-evidence") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            if (strcmp(value, "0") == 0) {
                config->min_induction_evidence = 0;
            } else if (!parse_size_text(value, &config->min_induction_evidence)) {
                return 0;
            }
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
                           "symbolic-transform-v10",
                           "certified_transform_gate",
                           "arc-transform-gate-v10",
                           &transform_metrics);
    print_arc_metrics_line("symbolic-transform-v10",
                           "certified_transform_gate",
                           &transform_metrics);
    arc_emit_builtin_program_summaries(&builtin_task, &config);
    nn_free(net);
    return EXIT_SUCCESS;
}
