#include "nn.h"
#include "nn_json.h"
#include "nn_holographic_ranker.h"
#include "holographic/leos_engine.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#define ARC_NOINLINE __attribute__((noinline))
#else
#define ARC_NOINLINE
#endif

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
    ARC_GRAMMAR_VERSION = 18,
    ARC_LEARNED_COMPOSE_TOP = 16,
    ARC_LEARNED_COMPOSE_LIMIT = 256,
    ARC_HYPOTHESIS_PROPOSAL_CAPACITY = 256,
    ARC_HYPOTHESIS_TOP_K = 32,
    ARC_HYPOTHESIS_STRUCTURAL_SCORE = 130,
    ARC_HYPOTHESIS_TRAIN_CONSISTENCY_PRIOR = 512,
    ARC_HYPOTHESIS_TEMPLATE_PRIOR_WEIGHT = 18,
    ARC_HYPOTHESIS_TEMPLATE_PROBE_WEIGHT = 10,
    ARC_HYPOTHESIS_SLOT_PROBE_WEIGHT = 14,
    ARC_HYPOTHESIS_LOG_ODDS_WEIGHT = 12,
    ARC_HYPOTHESIS_NEURAL_WEIGHT = 24,
    ARC_HYPOTHESIS_COLD_START_WEIGHT = 8,
    ARC_HYPOTHESIS_SLOT_PROBE_CAPACITY = 512,
    ARC_SKETCH_PROBE_INPUT_CAPACITY = 4096,
    ARC_SKETCH_PROBE_RECORD_CAPACITY = 4096,
    ARC_GENERATIVE_PRIOR_CAPACITY = 8192,
    ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY = 13,
    ARC_HYPOTHESIS_TEMPLATE_BUCKETS = 13,
    ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS = 9,
    ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS = 32,
    ARC_HYPOTHESIS_TEMPLATE_PRIOR_MIN_PROPOSALS = 1,
    ARC_HYPOTHESIS_PRIOR_MIN_PROPOSALS = 3,
    ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT = NN_HYPOTHESIS_FEATURE_COUNT,
    ARC_HYPOTHESIS_PRIOR_DEFAULT_HIDDEN = 24,
    ARC_HYPOTHESIS_PRIOR_EPOCHS = 32,
    ARC_HYPOTHESIS_TEMPLATE_SKETCH_PROGRAM_ID = 1036,
    ARC_PROGRAM_PROPOSER_BUDGET = 64,
    ARC_PROGRAM_BUDGET_GRID_COUNT = 4,
    ARC_PROGRAM_TASK_FEATURE_COUNT = 32,
    ARC_PROGRAM_FEATURE_COUNT = NN_PROGRAM_FEATURE_COUNT + ARC_PROGRAM_TASK_FEATURE_COUNT,
    ARC_PROGRAM_TRAIN_EPOCHS = 5,
    ARC_PROGRAM_STRATEGY_COUNT = 11,
    ARC_PROPOSAL_TRACE_NEGATIVE_CAP = 64,
    ARC_HOLO_BUCKET_GATE_FEATURE_COUNT = 8,
    ARC_HOLO_BUCKET_REVIEW_SELECTOR_BASE_FEATURES = 38,
    ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_HASH_BINS = 256,
    ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_FEATURES =
        ARC_HOLO_BUCKET_REVIEW_SELECTOR_BASE_FEATURES +
        ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_HASH_BINS,
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
    ARC_SEQUENCE_DIRECTION_HORIZONTAL = 0,
    ARC_SEQUENCE_DIRECTION_VERTICAL = 1,
    ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_RIGHT = 2,
    ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_LEFT = 3,
    ARC_SEQUENCE_DIRECTION_KIND_COUNT = 4,
    ARC_SEQUENCE_EXTEND_POSITIVE = 0,
    ARC_SEQUENCE_EXTEND_NEGATIVE = 1,
    ARC_SEQUENCE_EXTEND_BOTH = 2,
    ARC_SEQUENCE_EXTEND_KIND_COUNT = 3,
    ARC_OBJECT_TRANSLATE_RENDER_BLANK = 0,
    ARC_OBJECT_TRANSLATE_RENDER_COPY = 1,
    ARC_OBJECT_TRANSLATE_RENDER_MOVE = 2,
    ARC_OBJECT_TRANSLATE_RENDER_KIND_COUNT = 3,
};

static const size_t ARC_PROGRAM_BUDGET_GRID[ARC_PROGRAM_BUDGET_GRID_COUNT] = {
    8,
    16,
    32,
    64,
};

static const size_t ARC_PROGRAM_BUDGET_TIGHT_GRID[ARC_PROGRAM_BUDGET_GRID_COUNT] = {
    1,
    2,
    4,
    8,
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
    ARC_RULE_LAYER_OVERLAP = 32,
    ARC_RULE_LAYER_UNION = 33,
    ARC_RULE_LAYER_XOR = 34,
    ARC_RULE_LAYER_LEFT_ONLY = 35,
    ARC_RULE_LAYER_RIGHT_ONLY = 36,
    ARC_RULE_COMPRESS_COLOR_LINES = 37,
    ARC_RULE_TILE_MOTIF_PARITY_REPEAT = 38,
    ARC_RULE_GRAVITY_DOWN = 39,
    ARC_RULE_GRAVITY_UP = 40,
    ARC_RULE_GRAVITY_LEFT = 41,
    ARC_RULE_GRAVITY_RIGHT = 42,
    ARC_RULE_TILE_REPEAT = 43,
    ARC_RULE_GRID_REPEAT = 44,
    ARC_RULE_EXTRACT_COLOR = 45,
    ARC_RULE_ERASE_COLOR = 46,
    ARC_RULE_RECOLOR_BY_REFERENCE = 47,
    ARC_RULE_PATTERN_EXTEND = 48,
    ARC_RULE_DRAW_LINE = 49,
    ARC_RULE_SYMMETRY_H = 50,
    ARC_RULE_SYMMETRY_V = 51,
    ARC_RULE_MULTISCALE = 52,
    ARC_RULE_OBJECT_LABEL = 53,
    ARC_RULE_HOLO_TILE = 54,
    ARC_RULE_DILATE = 55,
    ARC_RULE_REFLECT_TILE = 56,
    ARC_RULE_MARKER_MOTIF_TILE = 57,
    ARC_RULE_AXIS_TILE = 58,
    ARC_RULE_PERIOD_EXTEND = 59,
    ARC_RULE_COUNT = 60,
} ARCTransformRule;

typedef enum {
    ARC_SKETCH_SELECTOR_GRID_PANEL = 0,
    ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL = 1,
    ARC_SKETCH_SELECTOR_COLOR_BBOX = 2,
    ARC_SKETCH_SELECTOR_LATTICE = 3,
} ARCSketchSelectorKind;

typedef enum {
    ARC_SKETCH_OP_CROP = 0,
    ARC_SKETCH_OP_TILE_REPEAT = 1,
    ARC_SKETCH_OP_MACROCELL_SUBSTITUTE = 2,
    ARC_SKETCH_OP_LATTICE_SAMPLE = 3,
    ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY = 4,
    ARC_SKETCH_OP_NORMALIZED_CROP = 5,
    ARC_SKETCH_OP_POINT_LINE_PROJECT = 6,
    ARC_SKETCH_OP_OBJECT_MASK_EXTRACT = 7,
    ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS = 8,
    ARC_SKETCH_OP_SEPARATOR_PANEL_MAP = 9,
    ARC_SKETCH_OP_PALETTE_SCALE = 10,
    ARC_SKETCH_OP_OBJECT_COLOR_TILE = 11,
    ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE = 12,
    ARC_SKETCH_OP_COMPONENT_COUNT_LINE = 13,
    ARC_SKETCH_OP_RECOLOR_CROP = 14,
    ARC_SKETCH_OP_MARKER_POLYLINE = 15,
    ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR = 16,
    ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK = 17,
    ARC_SKETCH_OP_VISUAL_COMBINATOR = 18,
    ARC_SKETCH_OP_EXTEND_ALIGNED_SEQUENCE = 19,
    ARC_SKETCH_OP_OBJECT_TRANSLATE = 20,
} ARCSketchOpKind;

typedef struct {
    int selector;
    int op;
    uint8_t background;
    uint8_t color;
    uint8_t output_color;
    int x_anchor;
    int y_anchor;
    int x_offset;
    int y_offset;
    size_t width;
    size_t height;
    size_t period;
} ARCSketchProgramSpec;

#define ARC_EXACT_SKETCH_PRIOR_CAPACITY 2048u
#define ARC_EXACT_SKETCH_PRIOR_SLOT_COUNT 12u
#define ARC_EXACT_SKETCH_PRIOR_COMPLETE_MASK \
    ((1u << ARC_EXACT_SKETCH_PRIOR_SLOT_COUNT) - 1u)

typedef struct {
    char library_id[ARC_TRANSFORM_NAME_BYTES * 2];
    char task_id[ARC_MAX_TASK_ID];
    ARCSketchProgramSpec spec;
    unsigned int seen_slot_mask;
    size_t proposal_count;
    size_t certified_count;
    double utility_mean;
} ARCExactSketchPrior;

typedef struct {
    int op;
    size_t input_width;
    size_t input_height;
    size_t output_width;
    size_t output_height;
    size_t shape_scale_x;
    size_t shape_scale_y;
    size_t input_component_count;
    size_t output_component_count;
    size_t accepted_count;
    size_t rejected_count;
    size_t near_miss_rejected_count;
    size_t hard_rejected_count;
    double mean_verifier_score;
    double rejection_penalty;
    double score;
} ARCGenerativePriorRow;

typedef enum {
    ARC_VISUAL_PIPELINE_CROP = 0,
    ARC_VISUAL_PIPELINE_TILE_REPEAT = 1,
    ARC_VISUAL_PIPELINE_MACROCELL_SUBSTITUTE = 2,
    ARC_VISUAL_PIPELINE_LATTICE_SAMPLE = 3,
    ARC_VISUAL_PIPELINE_COMPONENT_COLOR_SUMMARY = 4,
    ARC_VISUAL_PIPELINE_NORMALIZED_CROP = 5,
    ARC_VISUAL_PIPELINE_POINT_LINE_PROJECT = 6,
    ARC_VISUAL_PIPELINE_OBJECT_MASK_EXTRACT = 7,
    ARC_VISUAL_PIPELINE_AXIS_PROJECTION_COMPRESS = 8,
    ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_MAP = 9,
    ARC_VISUAL_PIPELINE_PALETTE_SCALE = 10,
    ARC_VISUAL_PIPELINE_OBJECT_COLOR_TILE = 11,
    ARC_VISUAL_PIPELINE_MARKER_COUNT_MOTIF_TILE = 12,
    ARC_VISUAL_PIPELINE_COMPONENT_COUNT_LINE = 13,
    ARC_VISUAL_PIPELINE_RECOLOR_CROP = 14,
    ARC_VISUAL_PIPELINE_MARKER_POLYLINE = 15,
    ARC_VISUAL_PIPELINE_SEPARATOR_INTERSECTION_COLOR = 16,
    ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_RELATION_MASK = 17,
    ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR = 18,
    ARC_VISUAL_PIPELINE_OBJECT_TRANSLATE = 19,
} ARCVisualPipelineKind;

typedef struct {
    ARCVisualPipelineKind kind;
    const char *composition_kind;
    const char *normalization_frame;
    const char *typed_stages;
    const char *primitive_contract;
    int selector;
    uint8_t background;
    uint8_t color;
    uint8_t output_color;
    uint8_t selector_kind;
    int include_background;
    int orientation;
    int count_mode;
    int x_align;
    int y_align;
    int x_anchor;
    int y_anchor;
    int x_offset;
    int y_offset;
    size_t width;
    size_t height;
    size_t period;
} ARCVisualPipelineSpec;

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
    .grounding_authority = NN_GROUNDING_EXACT_CHECK,
    .grounding_operator = "held-out ARC output grid for the fixed compiled task",
    .exact_verifier = "held-out output grid supplies the exact color for the requested cell",
    .verifier_feedback = "target color, top-k correctness, NLL, calibration bins, and proposal rank",
    .candidate_grammar = "ten output-color actions plus transform hypotheses certified on train pairs",
    .outcome_row_schema = "crlp.arc_transform.summary.v1 and fixed-cell trace rows",
    .claim_scope = "fixed-task output-cell scoring and grammar-consistent transform rows for ARC task 0520fde7",
    .known_non_claims =
        "does not claim public ARC-AGI leaderboard generality from the fixed-task exact-cell smoke",
    .rule_grammar =
        "copy_left, copy_right, mask_intersection, mask_union, object_select_overlap, "
        "layer_overlap, layer_union, layer_xor, layer_left_only, layer_right_only, "
        "color_map, same_grid_color_map, crop, compress_color_lines, tile_motif_parity_repeat, masked_motif_lift, rotate, flip, transpose, scale, bbox, translate, "
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

typedef enum {
    ARC_VSA_REPRESENTATION_GRID = 0,
    ARC_VSA_REPRESENTATION_OBJECT_SCENE = 1
} ARCVSARepresentation;

typedef struct {
    size_t epochs;
    size_t steps_per_epoch;
    size_t test_samples;
    size_t max_tasks;
    size_t program_budget;
    size_t program_train_max_tasks;
    size_t min_induction_evidence;
    size_t hypothesis_prior_hidden;
    size_t hypothesis_top_k;
    size_t arc_runtime_sketch_mine_cap;
    size_t arc_generative_prior_op_cap;
    uint64_t seed;
    nn_real learning_rate;
    nn_real program_learning_rate;
    const char *save_path;
    const char *task_dir;
    const char *program_train_dir;
    const char *hypothesis_library_prior_path;
    const char *save_hypothesis_prior_path;
    const char *vsa_ranker_path;
    ARCVSARepresentation vsa_representation;
    const char *arc_generative_prior_path;
    const char *task_ids;
    const char *exclude_task_ids;
    const char *arc_sketch_probe_candidates_path;
    const char *arc_sketch_probe_public_arc_dir;
    const char *arc_sketch_probe_sources_path;
    const char *arc_sketch_probe_prior_path;
    const char *arc_sketch_probe_attempts_path;
    const char *arc_proposal_trace_path;
    int loader_summary_only;
    int arc_holo_residual_diagnostics;
    int arc_holo_pressure_residual;
    size_t arc_holo_bucket_min_budget;
    size_t arc_holo_bucket_review_slots;
    size_t arc_holo_bucket_min_pressure_updates;
    int arc_holo_bucket_learned_gate;
    size_t arc_holo_bucket_learned_gate_min_examples;
    nn_real arc_holo_bucket_learned_gate_threshold;
    const char *arc_holo_bucket_review_selector_shadow_path;
    int shuffle_tasks;
    int unique_task_gate;
    int program_budget_grid;
    int induction_smoke;
    size_t arc_sketch_probe_sketches_per_task;
    int hypothesis_neural_prior_enabled;
    int hypothesis_slot_prior_enabled;
    int hypothesis_cold_start_enabled;
    int hypothesis_structural_score_enabled;
    int hypothesis_coverage_selection_enabled;
    int hypothesis_train_filter_enabled;
    int hypothesis_train_consistency_prior_enabled;
    int arc_visual_only;
    int disable_reflect_tile;
    int disable_marker_motif_tile;
    int disable_axis_tile;
    int disable_period_extend;
    const char *holo_prior_path;
    const char *save_learned_state_path;
    const char *load_learned_state_path;
    size_t task_epochs;
} ARCConfig;

static const ARCConfig *g_arc_active_config = NULL;

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
    char family[ARC_TRANSFORM_NAME_BYTES];
    char task_id[ARC_MAX_TASK_ID];
    char split[64];
    char op_hints[256];
    char coverage_frontier_id[ARC_TRANSFORM_NAME_BYTES];
    char parent_attempt_id[ARC_TRANSFORM_NAME_BYTES * 2];
    char repair_operator[ARC_TRANSFORM_NAME_BYTES];
    char dominant_residual_class[ARC_TRANSFORM_NAME_BYTES];
    char dominant_failure_stage[64];
    char repair_strategy[ARC_TRANSFORM_NAME_BYTES];
    size_t support_tasks;
    size_t proposal_count;
    double utility_mean;
} ARCSketchProbeInput;

typedef struct {
    char family[ARC_TRANSFORM_NAME_BYTES];
    char task_id[ARC_MAX_TASK_ID];
    char split[64];
    char task_path[ARC_MAX_TASK_PATH];
    char library_id[ARC_TRANSFORM_NAME_BYTES * 2];
    ARCSketchProgramSpec spec;
    char coverage_frontier_id[ARC_TRANSFORM_NAME_BYTES];
    char parent_attempt_id[ARC_TRANSFORM_NAME_BYTES * 2];
    char repair_operator[ARC_TRANSFORM_NAME_BYTES];
    char dominant_residual_class[ARC_TRANSFORM_NAME_BYTES];
    char dominant_failure_stage[64];
    char repair_strategy[ARC_TRANSFORM_NAME_BYTES];
    size_t support_tasks;
    size_t proposal_count;
    double utility_mean;
    size_t sketch_index;
} ARCSketchProbeRecord;

typedef struct {
    char family[ARC_TRANSFORM_NAME_BYTES];
    char task_id[ARC_MAX_TASK_ID];
    char split[64];
    char task_path[ARC_MAX_TASK_PATH];
    size_t attempted;
    size_t emitted;
    size_t pruned_infeasible;
    size_t unsupported_shape;
    size_t no_train_match;
    size_t duplicate;
    size_t cap_hit;
} ARCSketchProbeSummary;

typedef struct {
    size_t train_pairs;
    size_t checked_cells;
    size_t mismatches;
    size_t unsupported_pairs;
    size_t apply_failed_pairs;
    size_t shape_mismatch_pairs;
    size_t missing_output_pairs;
    size_t first_failing_pair;
    size_t first_input_width;
    size_t first_input_height;
    size_t first_expected_width;
    size_t first_expected_height;
    size_t first_produced_width;
    size_t first_produced_height;
    size_t first_pair_checked_cells;
    size_t first_pair_mismatches;
    size_t first_mismatch_min_x;
    size_t first_mismatch_min_y;
    size_t first_mismatch_max_x;
    size_t first_mismatch_max_y;
    double verifier_score;
} ARCTrainMatchTelemetry;

typedef struct {
    size_t input_component_count;
    size_t selected_component_count;
    size_t selected_component_area_min;
    size_t selected_component_area_max;
    size_t selected_component_area_total;
    size_t selected_component_shape_count;
    size_t selected_component_bbox_width;
    size_t selected_component_bbox_height;
    size_t output_foreground_cell_count;
    size_t output_component_count;
    size_t output_non_background_color_count;
    int selected_component_same_shape;
    int selected_count_matches_output_components;
    int selected_count_matches_output_foreground;
    int selected_area_matches_output_foreground;
    int selected_bbox_matches_output_shape;
    int selected_bbox_rotated_output_shape;
} ARCComponentInventoryFeatures;

typedef enum {
    ARC_SKETCH_PROBE_REASON_ACCEPTED = 0,
    ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE = 1,
    ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH = 2,
    ARC_SKETCH_PROBE_REASON_DUPLICATE = 3,
    ARC_SKETCH_PROBE_REASON_CAP_HIT = 4,
    ARC_SKETCH_PROBE_REASON_INVALID_ARGUMENT = 5,
} ARCSketchProbeReason;

typedef struct {
    char family[ARC_TRANSFORM_NAME_BYTES];
    char task_id[ARC_MAX_TASK_ID];
    char split[64];
    char task_path[ARC_MAX_TASK_PATH];
    ARCVisualPipelineSpec pipeline;
    ARCSketchProgramSpec spec;
    char coverage_frontier_id[ARC_TRANSFORM_NAME_BYTES];
    char parent_attempt_id[ARC_TRANSFORM_NAME_BYTES * 2];
    char repair_operator[ARC_TRANSFORM_NAME_BYTES];
    char dominant_residual_class[ARC_TRANSFORM_NAME_BYTES];
    char dominant_failure_stage[64];
    char repair_strategy[ARC_TRANSFORM_NAME_BYTES];
    size_t attempt_index;
    size_t support_tasks;
    size_t proposal_count;
    double utility_mean;
    int accepted;
    ARCSketchProbeReason reason;
    char failure_stage[64];
    char failure_detail[96];
    int has_contrast_accepted;
    size_t contrast_accepted_log_index;
    size_t contrast_slot_distance;
    size_t contrast_rejected_sibling_count;
    size_t input_width;
    size_t input_height;
    size_t output_width;
    size_t output_height;
    size_t shape_scale_x;
    size_t shape_scale_y;
    int period_divides_input_width;
    int period_divides_input_height;
    int slot_width_matches_output;
    int slot_height_matches_output;
    int color_in_input;
    int color_in_output;
    int output_color_in_output;
    int background_in_input;
    ARCComponentInventoryFeatures component_inventory;
    ARCTrainMatchTelemetry telemetry;
} ARCVisualPipelineAttempt;

typedef struct {
    ARCVisualPipelineAttempt *items;
    size_t count;
    size_t capacity;
} ARCVisualPipelineAttemptLog;

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
    ARC_PROGRAM_STRATEGY_TRACE_PRIOR = 4,
    ARC_PROGRAM_STRATEGY_TRACE_TASK_PRIOR = 5,
    ARC_PROGRAM_STRATEGY_TRACE_LINEAR = 6,
    ARC_PROGRAM_STRATEGY_TRACE_MLP = 7,
    ARC_PROGRAM_STRATEGY_TRACE_HOLO_RESIDUAL = 8,
    ARC_PROGRAM_STRATEGY_TRACE_HOLO_BUCKET = 9,
    ARC_PROGRAM_STRATEGY_ORACLE = 10,
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
    int enabled;
    size_t feature_count;
    size_t hash_bins;
    double weights[ARC_HOLO_BUCKET_REVIEW_SELECTOR_MAX_FEATURES];
} ARCHoloBucketReviewSelectorModel;

typedef struct {
    size_t family_seen[32];
    size_t family_success[32];
    size_t op_seen[ARC_RULE_COUNT];
    size_t op_success[ARC_RULE_COUNT];
    double family_task_feature_sum[32][ARC_PROGRAM_TASK_FEATURE_COUNT];
    double op_task_feature_sum[ARC_RULE_COUNT][ARC_PROGRAM_TASK_FEATURE_COUNT];
    double linear_weights[ARC_PROGRAM_FEATURE_COUNT];
    double linear_bias;
    size_t linear_updates;
    NeuralNet *mlp;
    size_t mlp_updates;
    HolographicRanker *holo_residual;
    size_t holo_updates;
    HoloPartitionedRanker *holo_pressure;
    size_t holo_pressure_updates;
    int holo_pressure_enabled;
    size_t holo_bucket_min_budget;
    size_t holo_bucket_review_slots;
    size_t holo_bucket_min_pressure_updates;
    int holo_bucket_learned_gate_enabled;
    size_t holo_bucket_learned_gate_min_examples;
    double holo_bucket_learned_gate_threshold;
    double holo_bucket_gate_weights[ARC_HOLO_BUCKET_GATE_FEATURE_COUNT];
    double holo_bucket_gate_bias;
    size_t holo_bucket_gate_updates;
    size_t holo_bucket_gate_positive_updates;
    double holo_bucket_gate_target_sum;
    double holo_bucket_gate_margin_sum;
    ARCHoloBucketReviewSelectorModel holo_bucket_review_selector_shadow;
    size_t input_color_seen[ARC_COLOR_COUNT];
    size_t input_color_success[ARC_COLOR_COUNT];
    size_t output_color_seen[ARC_COLOR_COUNT];
    size_t output_color_success[ARC_COLOR_COUNT];
    size_t background_color_seen[ARC_COLOR_COUNT];
    size_t background_color_success[ARC_COLOR_COUNT];
    size_t selector_seen[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t selector_success[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t output_selector_seen[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t output_selector_success[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t render_seen[ARC_OBJECT_RENDER_KIND_COUNT];
    size_t render_success[ARC_OBJECT_RENDER_KIND_COUNT];
    size_t scale_seen[8];
    size_t scale_success[8];
    size_t observed_tasks;
    size_t observed_rows;
    size_t positive_rows;
} ARCProposalTracePrior;

typedef struct {
    uint32_t template_id;
    char library_id[ARC_TRANSFORM_NAME_BYTES];
    char slot_name[NN_HYPOTHESIS_MAX_SLOT_NAME];
    NNHypothesisSlotType slot_type;
    double slot_value;
    int64_t slot_integer;
    size_t proposal_count;
} ARCNeuralSlotProbeRow;

typedef struct {
    NeuralNet *net;
    NeuralNet *hypothesis_prior_net;
    size_t trained_tasks;
    size_t train_updates;
    size_t hypothesis_prior_samples;
    size_t hypothesis_prior_updates;
    double hypothesis_prior_loss;
    size_t hypothesis_prior_hidden;
    size_t hypothesis_top_k;
    size_t arc_runtime_sketch_mine_cap;
    size_t arc_generative_prior_op_cap;
    int hypothesis_neural_prior_enabled;
    int hypothesis_slot_prior_enabled;
    int hypothesis_cold_start_enabled;
    int hypothesis_structural_score_enabled;
    int hypothesis_coverage_selection_enabled;
    int hypothesis_train_filter_enabled;
    int hypothesis_train_consistency_prior_enabled;
    int arc_visual_only;
    size_t hypothesis_library_entries;
    size_t hypothesis_library_template_rows;
    size_t hypothesis_library_slot_rows;
    size_t hypothesis_library_seeded_proposals;
    size_t hypothesis_library_seeded_certified;
    size_t hypothesis_probe_template_rows;
    size_t hypothesis_probe_seeded_proposals;
    size_t hypothesis_probe_slot_rows;
    size_t hypothesis_probe_slot_seeded_proposals;
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
    size_t hypothesis_template_proposal_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS];
    size_t hypothesis_template_certified_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS];
    size_t hypothesis_template_probe_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS];
    ARCNeuralSlotProbeRow hypothesis_slot_probe_rows[ARC_HYPOTHESIS_SLOT_PROBE_CAPACITY];
    size_t hypothesis_slot_probe_row_count;
    size_t hypothesis_slot_proposal_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS]
                                         [ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS]
                                         [ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS];
    size_t hypothesis_slot_certified_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS]
                                          [ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS]
                                          [ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS];
    size_t hypothesis_slot_probe_count[ARC_HYPOTHESIS_TEMPLATE_BUCKETS]
                                      [ARC_HYPOTHESIS_SLOT_TYPE_BUCKETS]
                                      [ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS];
    ARCExactSketchPrior exact_sketch_priors[ARC_EXACT_SKETCH_PRIOR_CAPACITY];
    size_t exact_sketch_prior_count;
    ARCGenerativePriorRow *generative_priors;
    size_t generative_prior_capacity;
    size_t generative_prior_count;
} ARCProgramLearnedState;

typedef struct ARCHypothesisTemplateBinding ARCHypothesisTemplateBinding;
typedef struct ARCHypothesisSelectionStats ARCHypothesisSelectionStats;

typedef const NNHypothesisTemplate *(*ARCHypothesisTemplateFn)(void);
typedef size_t (*ARCHypothesisInduceFn)(const ARCHypothesisTemplateBinding *binding,
                                        const ARCTask *task,
                                        ARCProgramLearnedState *learned,
                                        const ARCTransformCandidate *existing_candidates,
                                        size_t existing_count,
                                        ARCHypothesisProposal *proposals,
                                        size_t proposal_capacity,
                                        size_t proposal_count,
                                        ARCHypothesisSelectionStats *stats,
                                        size_t template_index);
typedef int (*ARCHypothesisToCandidateFn)(const NNHypothesis *hypothesis,
                                          ARCTransformCandidate *candidate);
typedef int (*ARCHypothesisApplyFn)(const NNHypothesis *hypothesis,
                                    const ARCGrid *input,
                                    ARCGrid *output);

struct ARCHypothesisSelectionStats {
    size_t registered_templates;
    size_t proposals_attempted;
    size_t proposals_retained;
    size_t proposals_generated;
    size_t proposals_train_consistent_pre_cap;
    size_t proposals_train_consistent;
    size_t proposals_duplicate_existing;
    size_t proposals_train_rejected;
    size_t proposals_heap_dropped;
    size_t proposals_heap_replaced;
    size_t proposals_selected;
    size_t proposals_selected_by_coverage;
    size_t proposals_selected_by_score_fallback;
    size_t runtime_sketch_mine_cap;
    size_t runtime_mined_sketches;
    size_t runtime_generative_prior_rows;
    size_t runtime_generative_prior_op_cap;
    size_t runtime_generative_prior_ops;
    size_t runtime_generative_prior_mined_sketches;
    size_t proposal_capacity;
    int proposal_cap_hit;
    int coverage_used;
    int coverage_fallback;
    size_t coverage_events;
    size_t hypothesis_prior_hidden;
    size_t hypothesis_top_k;
    int hypothesis_neural_prior_enabled;
    int hypothesis_slot_prior_enabled;
    int hypothesis_cold_start_enabled;
    int hypothesis_structural_score_enabled;
    int hypothesis_coverage_selection_enabled;
    int hypothesis_train_filter_enabled;
    int hypothesis_train_consistency_prior_enabled;
    int arc_visual_only;
    size_t library_prior_entries;
    size_t library_prior_template_rows;
    size_t library_prior_slot_rows;
    size_t library_prior_seeded_proposals;
    size_t library_prior_seeded_certified;
    size_t template_probe_rows;
    size_t template_probe_seeded_proposals;
    size_t slot_probe_rows;
    size_t slot_probe_seeded_proposals;
    size_t template_prior_proposals[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_prior_certified[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    double template_prior_score[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_probe_proposals[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    double template_probe_score[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    double slot_probe_score[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_attempted[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_generated[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_retained[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_train_consistent_pre_cap[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_train_consistent_retained[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_heap_dropped[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
    size_t template_selected[ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY];
};

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
static int arc_apply_multiscale_candidate(const ARCTransformCandidate *c, const ARCGrid *in, ARCGrid *out);
static int arc_grid_to_hologram(const ARCGrid *grid, double *scene_out);
static int arc_object_scene_to_hologram(const ARCGrid *grid, double *scene_out);
static int arc_proposal_trace_should_emit(const ARCProgramEval *eval,
                                          size_t rejected_emitted,
                                          size_t rejected_cap);

static int arc_apply_candidate(const ARCTransformCandidate *candidate,
                               const ARCGrid *input,
                               ARCGrid *output);
static int arc_candidate_matches_train(const ARCTask *t, const ARCTransformCandidate *c, size_t *ch, size_t *mm, size_t *up);
static int arc_build_tile_motif_parity_repeat_candidate(const ARCTask *task,
                                                        uint8_t background,
                                                        size_t period,
                                                        ARCTransformCandidate *candidate);
static int arc_apply_hypothesis_candidate(const ARCTransformCandidate *candidate,
                                          const ARCGrid *input,
                                          ARCGrid *output);
static const NNHypothesisTemplate *arc_hypothesis_template_same_grid_color_map(void);
static const NNHypothesisTemplate *arc_hypothesis_template_object_translate(void);
static const NNHypothesisTemplate *arc_hypothesis_template_move_to_anchor(void);
static const NNHypothesisTemplate *arc_hypothesis_template_recolor_by_reference(void);
static const NNHypothesisTemplate *arc_hypothesis_template_connect_aligned(void);
static const NNHypothesisTemplate *arc_hypothesis_template_fill_enclosed(void);
static const NNHypothesisTemplate *arc_hypothesis_template_object_extract(void);
static const NNHypothesisTemplate *arc_hypothesis_template_extend_aligned_sequence(void);
static const NNHypothesisTemplate *arc_hypothesis_template_tile_motif_repeat(void);
static const NNHypothesisTemplate *arc_hypothesis_template_masked_motif_lift(void);
static const NNHypothesisTemplate *arc_hypothesis_template_panel_selection(void);
static const NNHypothesisTemplate *arc_hypothesis_template_bbox_panel_selection(void);
static const NNHypothesisTemplate *arc_hypothesis_template_sketch_program(void);
static const ARCHypothesisTemplateBinding *arc_hypothesis_template_registry(size_t *count);
static const ARCHypothesisTemplateBinding *arc_hypothesis_binding_for_template_id(uint32_t template_id);
static int arc_hypothesis_from_candidate(const ARCTransformCandidate *candidate,
                                         NNHypothesis *hypothesis);
static int arc_candidate_from_color_map_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCTransformCandidate *candidate);
#include "arc_agi_hypothesis_templates.inc"

static HolographicRanker *g_vsa_ranker = NULL;
static ARCVSARepresentation g_vsa_representation = ARC_VSA_REPRESENTATION_GRID;
typedef struct {
    size_t score_attempts;
    size_t store_attempts;
    size_t stores;
    size_t blocked_no_ranker;
    size_t blocked_no_engine;
    size_t blocked_no_train;
    size_t blocked_hologram;
    size_t blocked_no_best_rule;
    size_t blocked_uncertified;
} ARCVSARankerStats;
static ARCVSARankerStats g_vsa_stats;

static const char *arc_vsa_representation_name(ARCVSARepresentation representation)
{
    switch (representation) {
    case ARC_VSA_REPRESENTATION_OBJECT_SCENE:
        return "arc_object_scene_v1";
    case ARC_VSA_REPRESENTATION_GRID:
    default:
        return "arc_grid_hologram_v1";
    }
}

static int arc_vsa_encode_grid(const ARCGrid *grid, double *scene_out)
{
    if (g_vsa_representation == ARC_VSA_REPRESENTATION_OBJECT_SCENE) {
        return arc_object_scene_to_hologram(grid, scene_out);
    }
    return arc_grid_to_hologram(grid, scene_out);
}

static ARCTransformMetrics evaluate_arc_transform_gate_candidates(const ARCTask *task, const ARCTransformCandidate *candidates, size_t candidate_count)
{
    ARCTransformMetrics metrics; static ARCGrid best[ARC_MAX_PAIRS];
    memset(&metrics,0,sizeof(metrics)); memset(best,0,sizeof(best));
    metrics.test_pairs=task->test_count; metrics.rules=candidate_count;
    double bs=-1e100; int hp=0; size_t best_rule=SIZE_MAX;
    double grid_hologram[256];
    int holo_ready = -1;
    static LeosEngine *vsa_eng = NULL;
    if (g_vsa_ranker == NULL) ++g_vsa_stats.blocked_no_ranker;
    if (g_vsa_ranker && !vsa_eng) { vsa_eng = leos_engine_create(256); }
    if (g_vsa_ranker && vsa_eng == NULL) ++g_vsa_stats.blocked_no_engine;
    if (g_vsa_ranker && vsa_eng && task->train_count > 0) {
        holo_ready = (arc_vsa_encode_grid(&task->train[0].input, grid_hologram) == 0) ? 1 : -1;
        if (holo_ready <= 0) ++g_vsa_stats.blocked_hologram;
    } else if (g_vsa_ranker && vsa_eng && task->train_count == 0) {
        ++g_vsa_stats.blocked_no_train;
    }
    for(size_t rule=0;rule<metrics.rules;++rule){
        size_t cc=0,mm=0,up=0;
        if(!arc_candidate_matches_train(task,&candidates[rule],&cc,&mm,&up)){metrics.train_checked_cells+=cc;metrics.train_mismatches+=mm;metrics.unsupported_train_pairs+=up;continue;}
        metrics.train_checked_cells+=cc;++metrics.train_consistent_rules;
        arc_append_accepted_rule(metrics.accepted_rules,sizeof(metrics.accepted_rules),candidates[rule].name);
        size_t ut=0; ARCGrid pr[ARC_MAX_PAIRS];
        if(!arc_apply_candidate_to_tests(task,&candidates[rule],pr,&ut)){metrics.unsupported_test_pairs+=ut;continue;}
        /* Blend structural score with holographic similarity when available */
        double sc = arc_structural_program_score(&candidates[rule]);
        if (g_vsa_ranker && vsa_eng && holo_ready > 0 && holo_ranker_count(g_vsa_ranker) > 0) {
            double op_vec[256];
            leos_engine_keygen(vsa_eng, (uint64_t)(candidates[rule].op * 1000 + (size_t)arc_candidate_family(&candidates[rule])), op_vec);
            double vsa_scores[1];
            ++g_vsa_stats.score_attempts;
            holo_ranker_score(g_vsa_ranker, grid_hologram, op_vec, 1, vsa_scores);
            sc += vsa_scores[0] * 50.0;
        }
        if(sc>bs){bs=sc;best_rule=rule;memcpy(best,pr,sizeof(best));hp=1;}
    }
    if(hp){memcpy(metrics.predictions,best,sizeof(metrics.predictions));metrics.prediction_count=task->test_count;}
    metrics.unique_test_prediction=1;
    int atk=task->test_count>0;
    for(size_t p=0;p<task->test_count;++p){if(!task->test[p].has_output){atk=0;continue;}size_t cells=arc_grid_cells(&task->test[p].output);metrics.test_cells+=cells;if(!hp||p>=metrics.prediction_count||!arc_grid_same_shape(&metrics.predictions[p],&task->test[p].output))continue;for(size_t c=0;c<cells;++c)metrics.test_correct_cells+=metrics.predictions[p].cells[c]==task->test[p].output.cells[c]?1:0;}
    metrics.test_exact=atk&&metrics.test_cells>0&&metrics.test_correct_cells==metrics.test_cells;
    metrics.certified=metrics.train_consistent_rules!=0;
    /* Store winning (grid, operator) pair in VSA ranker for future tasks */
    if (g_vsa_ranker) ++g_vsa_stats.store_attempts;
    if (g_vsa_ranker && !metrics.certified) ++g_vsa_stats.blocked_uncertified;
    if (g_vsa_ranker && metrics.certified && best_rule == SIZE_MAX) ++g_vsa_stats.blocked_no_best_rule;
    if (metrics.certified && best_rule != SIZE_MAX && g_vsa_ranker && vsa_eng && holo_ready > 0) {
        double op_vec[256];
        leos_engine_keygen(vsa_eng, (uint64_t)(candidates[best_rule].op * 1000 + (size_t)arc_candidate_family(&candidates[best_rule])), op_vec);
        holo_ranker_store(g_vsa_ranker, grid_hologram, op_vec);
        ++g_vsa_stats.stores;
    }
    return metrics;
}

/* Learned variant: uses family success statistics instead of structural score */
static ARCTransformMetrics evaluate_arc_transform_gate_candidates_learned(
    const ARCTask *task, const ARCTransformCandidate *candidates,
    size_t candidate_count, ARCProgramLearnedState *learned)
{
    ARCTransformMetrics metrics; static ARCGrid best[ARC_MAX_PAIRS];
    memset(&metrics,0,sizeof(metrics)); memset(best,0,sizeof(best));
    metrics.test_pairs=task->test_count; metrics.rules=candidate_count;
    double bs=-1e100; int hp=0; size_t best_rule=SIZE_MAX;
    double grid_hologram[256];
    int holo_ready = -1;
    static LeosEngine *vsa_eng = NULL;
    if (g_vsa_ranker == NULL) ++g_vsa_stats.blocked_no_ranker;
    if (g_vsa_ranker && !vsa_eng) { vsa_eng = leos_engine_create(256); }
    if (g_vsa_ranker && vsa_eng == NULL) ++g_vsa_stats.blocked_no_engine;
    if (g_vsa_ranker && vsa_eng && task->train_count > 0) {
        holo_ready = (arc_vsa_encode_grid(&task->train[0].input, grid_hologram) == 0) ? 1 : -1;
        if (holo_ready <= 0) ++g_vsa_stats.blocked_hologram;
    } else if (g_vsa_ranker && vsa_eng && task->train_count == 0) {
        ++g_vsa_stats.blocked_no_train;
    }
    for(size_t rule=0;rule<metrics.rules;++rule){
        size_t cc=0,mm=0,up=0;
        if(!arc_candidate_matches_train(task,&candidates[rule],&cc,&mm,&up)){metrics.train_checked_cells+=cc;metrics.train_mismatches+=mm;metrics.unsupported_train_pairs+=up;continue;}
        metrics.train_checked_cells+=cc;++metrics.train_consistent_rules;
        arc_append_accepted_rule(metrics.accepted_rules,sizeof(metrics.accepted_rules),candidates[rule].name);
        size_t ut=0; ARCGrid pr[ARC_MAX_PAIRS];
        if(!arc_apply_candidate_to_tests(task,&candidates[rule],pr,&ut)){metrics.unsupported_test_pairs+=ut;continue;}
        double sc = arc_structural_program_score(&candidates[rule]);
        if (learned) sc += arc_learned_family_score(learned, &candidates[rule]) * 10.0;
        if (g_vsa_ranker && vsa_eng && holo_ready > 0 && holo_ranker_count(g_vsa_ranker) > 0) {
            double op_vec[256];
            leos_engine_keygen(vsa_eng, (uint64_t)(candidates[rule].op * 1000 + (size_t)arc_candidate_family(&candidates[rule])), op_vec);
            double vsa_scores[1];
            ++g_vsa_stats.score_attempts;
            holo_ranker_score(g_vsa_ranker, grid_hologram, op_vec, 1, vsa_scores);
            sc += vsa_scores[0] * 50.0;
        }
        if(sc>bs){bs=sc;best_rule=rule;memcpy(best,pr,sizeof(best));hp=1;}
    }
    if(hp){memcpy(metrics.predictions,best,sizeof(metrics.predictions));metrics.prediction_count=task->test_count;}
    metrics.unique_test_prediction=1;
    int atk=task->test_count>0;
    for(size_t p=0;p<task->test_count;++p){if(!task->test[p].has_output){atk=0;continue;}size_t cells=arc_grid_cells(&task->test[p].output);metrics.test_cells+=cells;if(!hp||p>=metrics.prediction_count||!arc_grid_same_shape(&metrics.predictions[p],&task->test[p].output))continue;for(size_t c=0;c<cells;++c)metrics.test_correct_cells+=metrics.predictions[p].cells[c]==task->test[p].output.cells[c]?1:0;}
    metrics.test_exact=atk&&metrics.test_cells>0&&metrics.test_correct_cells==metrics.test_cells;
    metrics.certified=metrics.train_consistent_rules!=0;
    if (g_vsa_ranker) ++g_vsa_stats.store_attempts;
    if (g_vsa_ranker && !metrics.certified) ++g_vsa_stats.blocked_uncertified;
    if (g_vsa_ranker && metrics.certified && best_rule == SIZE_MAX) ++g_vsa_stats.blocked_no_best_rule;
    if (metrics.certified && best_rule != SIZE_MAX && g_vsa_ranker && vsa_eng && holo_ready > 0) {
        double op_vec[256];
        leos_engine_keygen(vsa_eng, (uint64_t)(candidates[best_rule].op * 1000 + (size_t)arc_candidate_family(&candidates[best_rule])), op_vec);
        holo_ranker_store(g_vsa_ranker, grid_hologram, op_vec);
        ++g_vsa_stats.stores;
    }
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
    nn_checkpoint_meta_init(&meta, 1u);
    if (!nn_checkpoint_meta_set_parent_model_hash(&meta, "unknown") ||
        !nn_checkpoint_meta_set_training_trace_root(&meta, "arc_agi_0520fde7") ||
        !nn_checkpoint_meta_set_notes(&meta, "fixed ARC-AGI-1 0520fde7 cell-color scorer") ||
        !nn_checkpoint_meta_set_feature_set(&meta, "arc-cell-color-v1")) {
        return 0;
    }
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

typedef NNJsonCursor ARCJsonParser;

static void print_json_string(const char *text)
{
    (void)nn_write_json_string(stdout, text);
}

static void arc_json_skip_ws(ARCJsonParser *parser)
{
    nn_json_skip_ws(parser);
}

static int arc_json_expect(ARCJsonParser *parser, char expected)
{
    return nn_json_expect(parser, expected);
}

static int arc_json_parse_string(ARCJsonParser *parser, char *out, size_t out_size)
{
    return nn_json_parse_string(parser, out, out_size);
}

static int arc_json_skip_value(ARCJsonParser *parser)
{
    return nn_json_skip_value(parser);
}

static int arc_json_parse_uint(ARCJsonParser *parser, unsigned long *out_value)
{
    return nn_json_parse_uint(parser, out_value);
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

#include "arc_agi_sketch_probes.inc"

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

static int arc_task_id_list_contains(const char *list, const char *task_id)
{
    if (list == NULL || list[0] == '\0' || task_id == NULL || task_id[0] == '\0') {
        return 0;
    }
    const char *cursor = list;
    while (*cursor != '\0') {
        while (*cursor == ',' || isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != ',') {
            ++cursor;
        }
        const char *end = cursor;
        while (end > start && isspace((unsigned char)end[-1])) {
            --end;
        }
        size_t len = (size_t)(end - start);
        if (len == strlen(task_id) && strncmp(start, task_id, len) == 0) {
            return 1;
        }
    }
    return 0;
}

static int arc_task_file_allowed(const ARCConfig *config, const char *task_id)
{
    if (config == NULL) {
        return 1;
    }
    if (config->task_ids != NULL &&
        !arc_task_id_list_contains(config->task_ids, task_id)) {
        return 0;
    }
    if (config->exclude_task_ids != NULL &&
        arc_task_id_list_contains(config->exclude_task_ids, task_id)) {
        return 0;
    }
    return 1;
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
                                                               1,
                                                               NULL)
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
    uint64_t prediction_hash = metrics != NULL && metrics->prediction_count != 0
                                   ? arc_prediction_hash(metrics->predictions,
                                                         metrics->prediction_count)
                                   : UINT64_C(0);
    printf("{\"schema\":\"crlp.arc_transform.summary.v1\",\"report_version\":1,"
           "\"task\":\"arc_agi\",\"task_instance\":");
    print_json_string(task_instance == NULL ? "0520fde7" : task_instance);
    printf(",\"grammar_version\":%d,\"rules\":%zu,\"train_consistent_rules\":%zu,"
           "\"train_checked_cells\":%zu,\"train_mismatches\":%zu,"
           "\"unsupported_train_pairs\":%zu,\"unsupported_test_pairs\":%zu,"
           "\"unique_test_prediction\":%s,\"test_exact\":%s,\"certified\":%s,"
           "\"test_pairs\":%zu,\"test_cells\":%zu,\"test_correct_cells\":%zu,"
           "\"prediction_hash\":\"%016llx\",\"accepted_rules\":",
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
           metrics->test_correct_cells,
           (unsigned long long)prediction_hash);
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

static void print_arc_vsa_ranker_summary_json(const char *path)
{
    printf("{\"schema\":\"crlp.arc_vsa_ranker.summary.v1\",\"report_version\":1,"
           "\"path\":");
    print_json_string(path);
    printf(",\"representation_condition\":");
    print_json_string(arc_vsa_representation_name(g_vsa_representation));
    printf(",\"score_attempts\":%zu,\"store_attempts\":%zu,"
           "\"stores\":%zu,\"stored_count\":%zu,\"fidelity\":%.8f,"
           "\"blocked_no_ranker\":%zu,\"blocked_no_engine\":%zu,"
           "\"blocked_no_train\":%zu,\"blocked_hologram\":%zu,"
           "\"blocked_no_best_rule\":%zu,\"blocked_uncertified\":%zu}\n",
           g_vsa_stats.score_attempts,
           g_vsa_stats.store_attempts,
           g_vsa_stats.stores,
           holo_ranker_count(g_vsa_ranker),
           holo_ranker_fidelity(g_vsa_ranker),
           g_vsa_stats.blocked_no_ranker,
           g_vsa_stats.blocked_no_engine,
           g_vsa_stats.blocked_no_train,
           g_vsa_stats.blocked_hologram,
           g_vsa_stats.blocked_no_best_rule,
           g_vsa_stats.blocked_uncertified);
}

static void print_arc_hypothesis_selection_json(const char *task_id,
                                                const ARCHypothesisSelectionStats *stats)
{
    if (stats == NULL) {
        return;
    }
    printf("{\"schema\":\"crlp.arc_hypothesis.selection.v1\",\"report_version\":1,"
           "\"task\":\"arc_agi\",\"task_instance\":");
    print_json_string(task_id == NULL ? "unknown" : task_id);
    printf(",\"grammar_version\":%d,"
           "\"coverage_mode\":\"test_input_predicted_change\","
           "\"top_k\":%zu,\"proposal_capacity\":%d,"
           "\"hypothesis_prior_hidden\":%zu,"
           "\"hypothesis_neural_prior_enabled\":%s,"
           "\"hypothesis_slot_prior_enabled\":%s,"
           "\"hypothesis_cold_start_enabled\":%s,"
           "\"hypothesis_structural_score_enabled\":%s,"
           "\"hypothesis_coverage_selection_enabled\":%s,"
           "\"hypothesis_train_filter_enabled\":%s,"
           "\"hypothesis_train_consistency_prior_enabled\":%s,"
           "\"arc_visual_only\":%s,"
           "\"score_weights\":{\"structural\":%d,\"train_consistency\":%d,\"log_odds\":%d,"
           "\"template_prior\":%d,\"template_probe\":%d,\"slot_probe\":%d,"
           "\"neural\":%d,\"cold_start\":%d},"
           "\"library_prior_entries\":%zu,"
           "\"library_prior_template_rows\":%zu,"
           "\"library_prior_slot_rows\":%zu,"
           "\"library_prior_seeded_proposals\":%zu,"
           "\"library_prior_seeded_certified\":%zu,"
           "\"template_probe_rows\":%zu,"
           "\"template_probe_seeded_proposals\":%zu,"
           "\"slot_probe_rows\":%zu,"
           "\"slot_probe_seeded_proposals\":%zu,"
           "\"registered_templates\":%zu,"
           "\"proposals_attempted\":%zu,"
           "\"proposals_retained\":%zu,"
           "\"proposals_generated\":%zu,"
           "\"proposals_train_consistent_pre_cap\":%zu,"
           "\"proposals_train_consistent\":%zu,"
           "\"proposals_duplicate_existing\":%zu,"
           "\"proposals_train_rejected\":%zu,"
           "\"proposals_heap_dropped\":%zu,"
           "\"proposals_heap_replaced\":%zu,"
           "\"proposals_selected\":%zu,"
           "\"proposals_selected_by_coverage\":%zu,"
           "\"proposals_selected_by_score_fallback\":%zu,"
           "\"runtime_sketch_mine_cap\":%zu,"
           "\"runtime_mined_sketches\":%zu,"
           "\"runtime_generative_prior_rows\":%zu,"
           "\"runtime_generative_prior_op_cap\":%zu,"
           "\"runtime_generative_prior_ops\":%zu,"
           "\"runtime_generative_prior_mined_sketches\":%zu,"
           "\"proposal_cap_hit\":%s,"
           "\"coverage_used\":%s,"
           "\"coverage_fallback\":%s,"
           "\"coverage_events\":%zu,"
           "\"templates\":[",
           ARC_GRAMMAR_VERSION,
           stats->hypothesis_top_k,
           ARC_HYPOTHESIS_PROPOSAL_CAPACITY,
           stats->hypothesis_prior_hidden,
           stats->hypothesis_neural_prior_enabled ? "true" : "false",
           stats->hypothesis_slot_prior_enabled ? "true" : "false",
           stats->hypothesis_cold_start_enabled ? "true" : "false",
           stats->hypothesis_structural_score_enabled ? "true" : "false",
           stats->hypothesis_coverage_selection_enabled ? "true" : "false",
           stats->hypothesis_train_filter_enabled ? "true" : "false",
           stats->hypothesis_train_consistency_prior_enabled ? "true" : "false",
           stats->arc_visual_only ? "true" : "false",
           stats->hypothesis_structural_score_enabled ? ARC_HYPOTHESIS_STRUCTURAL_SCORE : 0,
           stats->hypothesis_train_consistency_prior_enabled
               ? ARC_HYPOTHESIS_TRAIN_CONSISTENCY_PRIOR
               : 0,
           stats->hypothesis_slot_prior_enabled ? ARC_HYPOTHESIS_LOG_ODDS_WEIGHT : 0,
           stats->hypothesis_slot_prior_enabled ? ARC_HYPOTHESIS_TEMPLATE_PRIOR_WEIGHT : 0,
           ARC_HYPOTHESIS_TEMPLATE_PROBE_WEIGHT,
           ARC_HYPOTHESIS_SLOT_PROBE_WEIGHT,
           stats->hypothesis_neural_prior_enabled ? ARC_HYPOTHESIS_NEURAL_WEIGHT : 0,
           stats->hypothesis_cold_start_enabled ? ARC_HYPOTHESIS_COLD_START_WEIGHT : 0,
           stats->library_prior_entries,
           stats->library_prior_template_rows,
           stats->library_prior_slot_rows,
           stats->library_prior_seeded_proposals,
           stats->library_prior_seeded_certified,
           stats->template_probe_rows,
           stats->template_probe_seeded_proposals,
           stats->slot_probe_rows,
           stats->slot_probe_seeded_proposals,
           stats->registered_templates,
           stats->proposals_attempted,
           stats->proposals_retained,
           stats->proposals_generated,
           stats->proposals_train_consistent_pre_cap,
           stats->proposals_train_consistent,
           stats->proposals_duplicate_existing,
           stats->proposals_train_rejected,
           stats->proposals_heap_dropped,
           stats->proposals_heap_replaced,
           stats->proposals_selected,
           stats->proposals_selected_by_coverage,
           stats->proposals_selected_by_score_fallback,
           stats->runtime_sketch_mine_cap,
           stats->runtime_mined_sketches,
           stats->runtime_generative_prior_rows,
           stats->runtime_generative_prior_op_cap,
           stats->runtime_generative_prior_ops,
           stats->runtime_generative_prior_mined_sketches,
           stats->proposal_cap_hit ? "true" : "false",
           stats->coverage_used ? "true" : "false",
           stats->coverage_fallback ? "true" : "false",
           stats->coverage_events);
    size_t binding_count = 0;
    const ARCHypothesisTemplateBinding *bindings =
        arc_hypothesis_template_registry(&binding_count);
    int first = 1;
    for (size_t i = 0;
         i < binding_count && i < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY;
         ++i) {
        if (!first) {
            fputc(',', stdout);
        }
        first = 0;
        printf("{\"template_id\":%u,\"name\":",
               (unsigned)bindings[i].template_id);
        print_json_string(bindings[i].name);
        printf(",\"attempted\":%zu,"
               "\"generated\":%zu,"
               "\"retained\":%zu,"
               "\"template_prior_proposals\":%zu,"
               "\"template_prior_certified\":%zu,"
               "\"template_prior_score\":%.9g,"
               "\"template_probe_proposals\":%zu,"
               "\"template_probe_score\":%.9g,"
               "\"slot_probe_score\":%.9g,"
               "\"template_family_prior_source\":\"%s\","
               "\"train_consistent_pre_cap\":%zu,"
               "\"train_consistent_retained\":%zu,"
               "\"heap_dropped\":%zu,"
               "\"selected\":%zu}",
               stats->template_attempted[i],
               stats->template_generated[i],
               stats->template_retained[i],
               stats->template_prior_proposals[i],
               stats->template_prior_certified[i],
               stats->template_prior_score[i],
               stats->template_probe_proposals[i],
               stats->template_probe_score[i],
               stats->slot_probe_score[i],
               stats->template_prior_proposals[i] > 0
                   ? "library_prior"
                   : (stats->template_probe_proposals[i] > 0 ||
                              stats->slot_probe_score[i] > 0.0
                          ? "neural_probe"
                          : "none"),
               stats->template_train_consistent_pre_cap[i],
               stats->template_train_consistent_retained[i],
               stats->template_heap_dropped[i],
               stats->template_selected[i]);
    }
    printf("]}\n");
}

static void print_arc_program_task_audit_json(const ARCTask *task,
                                              const ARCTransformCandidate *candidates,
                                              const ARCProgramEval *evals,
                                              size_t candidate_count,
                                              const ARCHypothesisSelectionStats *stats)
{
    if (task == NULL || candidates == NULL || evals == NULL) {
        return;
    }
    size_t train_consistent = 0;
    size_t test_supported = 0;
    size_t exact = 0;
    size_t visual_train_consistent = 0;
    size_t visual_exact = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        int visual = arc_candidate_is_visual_program_hypothesis(&candidates[i]);
        if (evals[i].train_consistent) {
            ++train_consistent;
            if (visual) {
                ++visual_train_consistent;
            }
        }
        if (evals[i].test_supported) {
            ++test_supported;
        }
        if (arc_program_eval_is_exact_success(&evals[i])) {
            ++exact;
            if (visual) {
                ++visual_exact;
            }
        }
    }
    printf("{\"schema\":\"crlp.arc_program_task.audit.v1\","
           "\"report_version\":1,\"task\":\"arc_agi\",\"task_instance\":");
    print_json_string(task->id);
    printf(",\"arc_visual_only\":%s,"
           "\"candidate_count\":%zu,"
           "\"train_consistent_candidates\":%zu,"
           "\"test_supported_candidates\":%zu,"
           "\"exact_candidates\":%zu,"
           "\"visual_train_consistent_candidates\":%zu,"
           "\"visual_exact_candidates\":%zu,"
           "\"top_exact_candidates\":[",
           stats != NULL && stats->arc_visual_only ? "true" : "false",
           candidate_count,
           train_consistent,
           test_supported,
           exact,
           visual_train_consistent,
           visual_exact);
    size_t emitted = 0;
    for (size_t i = 0; i < candidate_count && emitted < 8u; ++i) {
        if (!arc_program_eval_is_exact_success(&evals[i])) {
            continue;
        }
        if (emitted != 0) {
            fputc(',', stdout);
        }
        printf("{\"candidate_index\":%zu,\"name\":", i);
        print_json_string(candidates[i].name);
        printf(",\"op\":%d,"
               "\"hypothesis_template_id\":%u,"
               "\"visual_program\":%s,"
               "\"test_correct_cells\":%zu,"
               "\"test_cells\":%zu}",
               (int)candidates[i].op,
               (unsigned)candidates[i].hypothesis_template_id,
               arc_candidate_is_visual_program_hypothesis(&candidates[i])
                   ? "true"
                   : "false",
               evals[i].test_correct_cells,
               evals[i].test_cells);
        ++emitted;
    }
    printf("]}\n");
}

static void arc_emit_program_task_features_json(FILE *fp,
                                                const ARCProgramTaskFeatures *features)
{
    fprintf(fp,
            "{\"input_width\":%.9g,\"input_height\":%.9g,"
            "\"output_width\":%.9g,\"output_height\":%.9g,"
            "\"width_ratio\":%.9g,\"height_ratio\":%.9g,"
            "\"area_ratio\":%.9g,\"same_shape_fraction\":%.9g,"
            "\"output_smaller_fraction\":%.9g,"
            "\"output_larger_fraction\":%.9g,"
            "\"input_color_count\":%.9g,\"output_color_count\":%.9g,"
            "\"input_foreground_fraction\":%.9g,"
            "\"output_foreground_fraction\":%.9g,"
            "\"input_component_count\":%.9g,"
            "\"output_component_count\":%.9g,"
            "\"input_bbox_width_fraction\":%.9g,"
            "\"input_bbox_height_fraction\":%.9g,"
            "\"output_bbox_width_fraction\":%.9g,"
            "\"output_bbox_height_fraction\":%.9g,"
            "\"bbox_area_ratio\":%.9g,\"component_delta\":%.9g,"
            "\"color_delta\":%.9g,\"foreground_delta\":%.9g,"
            "\"input_object_count\":%.9g,\"output_object_count\":%.9g,"
            "\"input_repeated_shape_pairs\":%.9g,"
            "\"output_repeated_shape_pairs\":%.9g,"
            "\"input_aligned_pairs\":%.9g,"
            "\"output_aligned_pairs\":%.9g,"
            "\"input_adjacent_pairs\":%.9g,"
            "\"output_adjacent_pairs\":%.9g}",
            features->input_width,
            features->input_height,
            features->output_width,
            features->output_height,
            features->width_ratio,
            features->height_ratio,
            features->area_ratio,
            features->same_shape_fraction,
            features->output_smaller_fraction,
            features->output_larger_fraction,
            features->input_color_count,
            features->output_color_count,
            features->input_foreground_fraction,
            features->output_foreground_fraction,
            features->input_component_count,
            features->output_component_count,
            features->input_bbox_width_fraction,
            features->input_bbox_height_fraction,
            features->output_bbox_width_fraction,
            features->output_bbox_height_fraction,
            features->bbox_area_ratio,
            features->component_delta,
            features->color_delta,
            features->foreground_delta,
            features->input_object_count,
            features->output_object_count,
            features->input_repeated_shape_pairs,
            features->output_repeated_shape_pairs,
            features->input_aligned_pairs,
            features->output_aligned_pairs,
            features->input_adjacent_pairs,
            features->output_adjacent_pairs);
}

static double arc_program_eval_cell_accuracy(const ARCProgramEval *eval)
{
    if (eval == NULL || eval->test_cells == 0) {
        return 0.0;
    }
    return (double)eval->test_correct_cells / (double)eval->test_cells;
}

static int arc_proposal_trace_should_emit(const ARCProgramEval *eval,
                                          size_t rejected_emitted,
                                          size_t rejected_cap)
{
    if (eval == NULL) {
        return 0;
    }
    if (eval->train_consistent || arc_program_eval_is_exact_success(eval)) {
        return 1;
    }
    return rejected_emitted < rejected_cap;
}

static void arc_emit_proposal_trace_rows(FILE *fp,
                                         const ARCTask *task,
                                         const ARCTransformCandidate *candidates,
                                         const ARCProgramEval *evals,
                                         size_t candidate_count,
                                         size_t rejected_cap)
{
    if (fp == NULL || task == NULL || candidates == NULL || evals == NULL) {
        return;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    size_t rejected_emitted = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        int exact = arc_program_eval_is_exact_success(&evals[i]);
        if (!arc_proposal_trace_should_emit(&evals[i], rejected_emitted, rejected_cap)) {
            continue;
        }
        if (!evals[i].train_consistent && !exact) {
            ++rejected_emitted;
        }

        const ARCTransformCandidate *candidate = &candidates[i];
        fprintf(fp,
                "{\"schema\":\"crlp.arc_proposal_trace.v1\","
                "\"report_version\":1,"
                "\"task\":\"arc_agi\","
                "\"task_instance\":");
        nn_write_json_string(fp, task->id);
        fprintf(fp,
                ",\"candidate_index\":%zu,"
                "\"candidate_name\":",
                i);
        nn_write_json_string(fp, candidate->name);
        fprintf(fp,
                ",\"op\":%d,"
                "\"family\":%zu,"
                "\"structural_score\":%.9g,"
                "\"task_conditioned_score\":%.9g,"
                "\"composed\":%s,"
                "\"object_program\":%s,"
                "\"visual_program\":%s,"
                "\"hypothesis_induced\":%s,"
                "\"slots\":{\"input_color\":%u,\"output_color\":%u,"
                "\"background_color\":%u,\"selector_kind\":%u,"
                "\"output_selector_kind\":%u,\"reference_color\":%u,"
                "\"render_mode\":%u,\"dx\":%d,\"dy\":%d,\"scale\":%zu,"
                "\"hypothesis_template_id\":%u,"
                "\"hypothesis_slot_count\":%u,"
                "\"hypothesis_slot_integers\":[",
                (int)candidate->op,
                arc_candidate_family(candidate),
                arc_structural_program_score(candidate),
                arc_task_conditioned_program_score(task, candidate),
                arc_candidate_is_composed(candidate) ? "true" : "false",
                evals[i].object_program ? "true" : "false",
                arc_candidate_is_visual_program_hypothesis(candidate) ? "true" : "false",
                arc_candidate_is_hypothesis_induced(candidate) ? "true" : "false",
                (unsigned)candidate->input_color,
                (unsigned)candidate->output_color,
                (unsigned)candidate->background_color,
                (unsigned)candidate->selector_kind,
                (unsigned)candidate->output_selector_kind,
                (unsigned)candidate->reference_color,
                (unsigned)candidate->render_mode,
                candidate->dx,
                candidate->dy,
                candidate->scale,
                (unsigned)candidate->hypothesis_template_id,
                (unsigned)candidate->hypothesis_slot_count);
        size_t slot_count = candidate->hypothesis_slot_count;
        if (slot_count > NN_HYPOTHESIS_MAX_SLOTS) {
            slot_count = NN_HYPOTHESIS_MAX_SLOTS;
        }
        for (size_t slot = 0; slot < slot_count; ++slot) {
            if (slot != 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%lld", (long long)candidate->hypothesis_slot_integers[slot]);
        }
        fprintf(fp, "]},\"task_features\":");
        arc_emit_program_task_features_json(fp, &task_features);
        fprintf(fp,
                ",\"verifier\":{\"train_consistent\":%s,"
                "\"test_supported\":%s,\"test_exact\":%s,"
                "\"train_checked_cells\":%zu,"
                "\"train_mismatches\":%zu,"
                "\"unsupported_train_pairs\":%zu,"
                "\"unsupported_test_pairs\":%zu,"
                "\"test_correct_cells\":%zu,"
                "\"test_cells\":%zu,"
                "\"test_cell_accuracy\":%.9g,"
                "\"prediction_hash\":\"%016llx\","
                "\"program_hash\":\"%016llx\"}}\n",
                evals[i].train_consistent ? "true" : "false",
                evals[i].test_supported ? "true" : "false",
                exact ? "true" : "false",
                evals[i].train_checked_cells,
                evals[i].train_mismatches,
                evals[i].unsupported_train_pairs,
                evals[i].unsupported_test_pairs,
                evals[i].test_correct_cells,
                evals[i].test_cells,
                arc_program_eval_cell_accuracy(&evals[i]),
                (unsigned long long)evals[i].prediction_hash,
                (unsigned long long)evals[i].program_hash);
    }
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


static void arc_rotate_grid_cw(const ARCGrid *src, ARCGrid *dst) {
    dst->width = src->height; dst->height = src->width;
    for (size_t y = 0; y < src->height; ++y) for (size_t x = 0; x < src->width; ++x)
        dst->cells[(src->width - 1u - x) * dst->width + y] = src->cells[y * src->width + x];
}
static void arc_flip_grid_h(const ARCGrid *src, ARCGrid *dst) {
    dst->width = src->width; dst->height = src->height;
    for (size_t y = 0; y < src->height; ++y) for (size_t x = 0; x < src->width; ++x)
        dst->cells[y * dst->width + (src->width - 1u - x)] = src->cells[y * src->width + x];
}
static void arc_transform_task(const ARCTask *src, ARCTask *dst, int rotations, int flip_h) {
    memcpy(dst->id, src->id, sizeof(dst->id)); dst->train_count = src->train_count; dst->test_count = src->test_count;
    for (size_t pair = 0; pair < src->train_count; ++pair) {
        ARCGrid ci = src->train[pair].input, co; int ho = src->train[pair].has_output; if(ho)co=src->train[pair].output;
        if(flip_h){ARCGrid t=ci;arc_flip_grid_h(&t,&ci);if(ho){t=co;arc_flip_grid_h(&t,&co);}}
        for(int r=0;r<rotations;++r){ARCGrid ti=ci;arc_rotate_grid_cw(&ti,&ci);if(ho){ARCGrid to=co;arc_rotate_grid_cw(&to,&co);}}
        dst->train[pair].input=ci;dst->train[pair].output=co;dst->train[pair].has_output=ho;
    }
    for (size_t pair = 0; pair < src->test_count; ++pair) {
        ARCGrid ci=src->test[pair].input,co;int ho=src->test[pair].has_output;if(ho)co=src->test[pair].output;
        if(flip_h){ARCGrid t=ci;arc_flip_grid_h(&t,&ci);if(ho){t=co;arc_flip_grid_h(&t,&co);}}
        for(int r=0;r<rotations;++r){ARCGrid ti=ci;arc_rotate_grid_cw(&ti,&ci);if(ho){ARCGrid to=co;arc_rotate_grid_cw(&to,&co);}}
        dst->test[pair].input=ci;if(ho){dst->test[pair].output=co;dst->test[pair].has_output=1;}
    }
}

static int ARC_NOINLINE run_arc_task_loader(const ARCConfig *config)
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
    g_arc_active_config = config;
    g_vsa_representation = config->vsa_representation;
    if (config->vsa_ranker_path != NULL) { memset(&g_vsa_stats, 0, sizeof(g_vsa_stats)); g_vsa_ranker = holo_ranker_create(256); if (g_vsa_ranker) { FILE *tf = fopen(config->vsa_ranker_path, "rb"); if (tf) { fclose(tf); holo_ranker_load(g_vsa_ranker, config->vsa_ranker_path); } } }
    ARCProgramLearnedState learned_state;
    if (!arc_program_learned_init(&learned_state,
                                  config,
                                  (unsigned int)(config->seed & UINT32_MAX))) {
        free(files);
        return 0;
    }
    ARCProposalTracePrior proposal_trace_prior;
    if (!arc_proposal_trace_prior_init(&proposal_trace_prior,
                                       (unsigned int)(config->seed & UINT32_MAX))) {
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
    }
    proposal_trace_prior.holo_pressure_enabled = config->arc_holo_pressure_residual;
    proposal_trace_prior.holo_bucket_min_budget = config->arc_holo_bucket_min_budget;
    proposal_trace_prior.holo_bucket_review_slots =
        config->arc_holo_bucket_review_slots;
    proposal_trace_prior.holo_bucket_min_pressure_updates =
        config->arc_holo_bucket_min_pressure_updates;
    proposal_trace_prior.holo_bucket_learned_gate_enabled =
        config->arc_holo_bucket_learned_gate;
    proposal_trace_prior.holo_bucket_learned_gate_min_examples =
        config->arc_holo_bucket_learned_gate_min_examples;
    proposal_trace_prior.holo_bucket_learned_gate_threshold =
        (double)config->arc_holo_bucket_learned_gate_threshold;
    ARCTransformCandidate *candidates =
        calloc(ARC_RULE_CANDIDATE_COUNT, sizeof(*candidates));
    ARCProgramEval *evals = calloc(ARC_RULE_CANDIDATE_COUNT, sizeof(*evals));
    ARCTransformCandidate *learn_candidates =
        calloc(ARC_RULE_CANDIDATE_COUNT, sizeof(*learn_candidates));
    ARCProgramEval *learn_evals =
        calloc(ARC_RULE_CANDIDATE_COUNT, sizeof(*learn_evals));
    if (candidates == NULL || evals == NULL ||
        learn_candidates == NULL || learn_evals == NULL) {
        free(candidates);
        free(evals);
        free(learn_candidates);
        free(learn_evals);
        arc_proposal_trace_prior_free(&proposal_trace_prior);
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
    }
    if (config->arc_holo_bucket_review_selector_shadow_path != NULL &&
        !arc_holo_bucket_review_selector_load(
            &proposal_trace_prior.holo_bucket_review_selector_shadow,
            config->arc_holo_bucket_review_selector_shadow_path)) {
        fprintf(stderr,
                "Failed to load ARC Holo bucket review selector shadow model from %s\n",
                config->arc_holo_bucket_review_selector_shadow_path);
        free(candidates);
        free(evals);
        free(learn_candidates);
        free(learn_evals);
        arc_proposal_trace_prior_free(&proposal_trace_prior);
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
    }
    if (config->program_train_dir != NULL &&
        !arc_pretrain_program_proposer(&learned_state,
                                       config->program_train_dir,
                                       config->program_train_max_tasks,
                                       config->program_learning_rate,
                                       config->induction_smoke)) {
        free(candidates);
        free(evals);
        free(learn_candidates);
        free(learn_evals);
        arc_proposal_trace_prior_free(&proposal_trace_prior);
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
    }
    if (config->hypothesis_library_prior_path != NULL &&
        !arc_program_learned_load_hypothesis_library_prior(
            &learned_state,
            config->hypothesis_library_prior_path)) {
        free(candidates);
        free(evals);
        free(learn_candidates);
        free(learn_evals);
        arc_proposal_trace_prior_free(&proposal_trace_prior);
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
    }
    if (config->load_learned_state_path != NULL) {
        if (!arc_program_learned_load_state(&learned_state, config->load_learned_state_path)) {
            fprintf(stderr, "Failed to load learned state from %s\n", config->load_learned_state_path);
        }
    }
    if (config->arc_generative_prior_path != NULL &&
        !arc_program_learned_load_arc_generative_prior(
            &learned_state,
            config->arc_generative_prior_path)) {
        free(candidates);
        free(evals);
        free(learn_candidates);
        free(learn_evals);
        arc_proposal_trace_prior_free(&proposal_trace_prior);
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
    FILE *proposal_trace_fp = NULL;
    if (config->arc_proposal_trace_path != NULL) {
        proposal_trace_fp = fopen(config->arc_proposal_trace_path, "w");
        if (proposal_trace_fp == NULL) {
            fprintf(stderr,
                    "failed to open ARC proposal trace path: %s\n",
                    config->arc_proposal_trace_path);
            free(candidates);
            free(evals);
            free(learn_candidates);
            free(learn_evals);
            arc_proposal_trace_prior_free(&proposal_trace_prior);
            arc_program_learned_free(&learned_state);
            free(files);
            return 0;
        }
    }

    if (config->shuffle_tasks) { for(size_t i=0;i<file_count;++i){size_t j=i+(size_t)(arc_hash_mix((uint64_t)i,(uint64_t)config->seed)%(file_count-i));if(j!=i){ARCTaskFile t=files[i];files[i]=files[j];files[j]=t;}} }
    size_t induction_evidence_rows = 0;
    for (size_t epoch = 0; epoch < config->task_epochs; ++epoch) {
    for (size_t i = 0; i < file_count; ++i) {
        if (!arc_task_file_allowed(config, files[i].id)) {
            continue;
        }
        if (config->max_tasks != 0 && epoch == 0 && summary.loaded_tasks >= config->max_tasks) {
            break;
        }
        ARCTask task;
        if (!arc_load_task_file(files[i].path, files[i].id, &task)) {
            ++summary.parse_errors;
            continue;
        }
        ARCHypothesisSelectionStats hypothesis_stats;
        
        static uint8_t *solved_kinds = NULL;
        static size_t solved_kind_count = 0;
        if (config->holo_prior_path != NULL && solved_kinds == NULL) {
            solved_kinds = calloc(256, sizeof(uint8_t));
        }

        /* Cross-task holo cache: try transforms from similar previously-solved tasks */
        static ARCTransformCandidate *solved_xforms = NULL;
        static size_t solved_count = 0;
        if (config->holo_prior_path != NULL && solved_xforms == NULL) {
            solved_xforms = calloc(ARC_RULE_CANDIDATE_COUNT, sizeof(ARCTransformCandidate));
        }
        
        /* Try cached transforms first */
        ARCTask *rotated = calloc(1, sizeof(ARCTask));
        if (rotated == NULL) break;
        int aug_limit = config->unique_task_gate ? 1 : 8;
        for (int aug = 0; aug < aug_limit; ++aug) {
            int r = aug % 4, flip = aug >= 4 ? 1 : 0;
            if (aug == 0) *rotated = task; else arc_transform_task(&task, rotated, r, flip);
        int cache_hit = 0;
        for (size_t ci = 0; ci < solved_count && !cache_hit; ci++) {
            ARCTransformCandidate tc[1] = {solved_xforms[ci]};
            ARCTransformMetrics tm = evaluate_arc_transform_gate_candidates_learned(rotated, tc, 1, &learned_state);
            if (tm.certified) {
                arc_loader_summary_add(&summary, rotated, &tm);
                cache_hit = 1;
            }
        }
        if (cache_hit) {
            continue;
        }
        
        size_t candidate_count = arc_build_candidate_set(rotated,
                                                         &learned_state,
                                                         candidates,
                                                         ARC_RULE_CANDIDATE_COUNT,
                                                         config->induction_smoke,
                                                         &hypothesis_stats);
        
        /* Reorder candidates: prioritize transform types that solved similar tasks */
        if (config->holo_prior_path != NULL && solved_kind_count > 0) {
            ARCTransformCandidate reordered[ARC_RULE_CANDIDATE_COUNT];
            size_t ri = 0;
            /* First pass: candidates matching previously-successful transform types */
            for (size_t k = 0; k < solved_kind_count && k < 10; k++) {
                for (size_t j = 0; j < candidate_count; j++) {
                    if (candidates[j].output_selector_kind == solved_kinds[k]) {
                        reordered[ri++] = candidates[j];
                    }
                }
            }
            /* Second pass: remaining candidates */
            for (size_t j = 0; j < candidate_count; j++) {
                int already = 0;
                for (size_t r = 0; r < ri; r++) {
                    if (memcmp(&reordered[r], &candidates[j], sizeof(ARCTransformCandidate)) == 0)
                        { already = 1; break; }
                }
                if (!already) reordered[ri++] = candidates[j];
            }
            memcpy(candidates, reordered, ri * sizeof(ARCTransformCandidate));
        }
        
        ARCTransformMetrics transform =
            evaluate_arc_transform_gate_candidates_learned(rotated, candidates, candidate_count, &learned_state);
        arc_loader_summary_add(&summary, rotated, &transform);
        arc_evaluate_program_candidates(rotated, candidates, candidate_count, evals);
        arc_emit_proposal_trace_rows(proposal_trace_fp,
                                     rotated,
                                     candidates,
                                     evals,
                                     candidate_count,
                                     ARC_PROPOSAL_TRACE_NEGATIVE_CAP);
        
        /* Cache successful transform TYPE for cross-task reuse */
        if (transform.certified && solved_kinds && solved_kind_count < 200) {
            for (size_t j = 0; j < candidate_count; j++) {
                if (candidates[j].output_selector_kind != 0) {
                    uint8_t kind = candidates[j].output_selector_kind;
                    int seen = 0;
                    for (size_t k = 0; k < solved_kind_count; k++)
                        if (solved_kinds[k] == kind) { seen = 1; break; }
                    if (!seen) solved_kinds[solved_kind_count++] = kind;
                    break;
                }
            }
        }
        if (config->induction_smoke) {
            print_arc_hypothesis_selection_json(rotated->id, &hypothesis_stats);
            print_arc_program_task_audit_json(rotated,
                                              candidates,
                                              evals,
                                              candidate_count,
                                              &hypothesis_stats);
            induction_evidence_rows +=
                arc_emit_induction_smoke(rotated, candidates, evals, candidate_count);
        }
        for (size_t budget_index = 0; budget_index < budget_count; ++budget_index) {
            if (config->arc_holo_residual_diagnostics) {
                arc_emit_holo_residual_diagnostics(&task,
                                                   candidates,
                                                   evals,
                                                   candidate_count,
                                                   &learned_state,
                                                   &proposal_trace_prior,
                                                   budgets[budget_index]);
                arc_emit_holo_bucket_gate_diagnostic(&task,
                                                     candidates,
                                                     evals,
                                                     candidate_count,
                                                     &proposal_trace_prior,
                                                     budgets[budget_index]);
                arc_emit_holo_bucket_diagnostics(&task,
                                                 candidates,
                                                 evals,
                                                 candidate_count,
                                                 &learned_state,
                                                 &proposal_trace_prior,
                                                 budgets[budget_index]);
            }
            for (size_t strategy = 0; strategy < ARC_PROGRAM_STRATEGY_COUNT; ++strategy) {
                arc_program_aggregate_task(&program_aggregates[budget_index][strategy],
                                           &task,
                                           candidates,
                                           evals,
                                           candidate_count,
                                           (ARCProgramStrategy)strategy,
                                           &learned_state,
                                           &proposal_trace_prior,
                                           budgets[budget_index]);
            }
        }
        arc_proposal_trace_prior_observe_task(&proposal_trace_prior,
                                              rotated,
                                              candidates,
                                              evals,
                                              candidate_count,
                                              budgets,
                                              budget_count,
                                              ARC_PROPOSAL_TRACE_NEGATIVE_CAP);
        if (config->program_train_dir == NULL) {
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
                free(candidates);
                free(evals);
                free(learn_candidates);
                free(learn_evals);
                if (proposal_trace_fp != NULL) {
                    fclose(proposal_trace_fp);
                }
                arc_proposal_trace_prior_free(&proposal_trace_prior);
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
                                      "symbolic-transform-v14",
                                       "certified_transform_gate",
                                      "arc-transform-gate-v14",
                                       &transform_metrics);
            }
        }
        } free(rotated);
    }
    }
    if (config->induction_smoke &&
        induction_evidence_rows < config->min_induction_evidence) {
        fprintf(stderr,
                "ARC induction emitted %zu hypothesis evidence rows; expected at least %zu\n",
                induction_evidence_rows,
                config->min_induction_evidence);
        if (proposal_trace_fp != NULL) {
            fclose(proposal_trace_fp);
        }
        free(candidates);
        free(evals);
        free(learn_candidates);
        free(learn_evals);
        arc_proposal_trace_prior_free(&proposal_trace_prior);
        arc_program_learned_free(&learned_state);
        free(files);
        return 0;
    }
    if (config->save_hypothesis_prior_path != NULL) arc_program_learned_save_hypothesis_library_prior(&learned_state, config->save_hypothesis_prior_path);
    if (config->vsa_ranker_path != NULL && g_vsa_ranker != NULL) holo_ranker_save(g_vsa_ranker, config->vsa_ranker_path);
    if (config->vsa_ranker_path != NULL) print_arc_vsa_ranker_summary_json(config->vsa_ranker_path);
    if (proposal_trace_fp != NULL) {
        fclose(proposal_trace_fp);
    }
    print_arc_loader_summary_json(config->task_dir, &summary);
    for (size_t budget_index = 0; budget_index < budget_count; ++budget_index) {
        for (size_t strategy = 0; strategy < ARC_PROGRAM_STRATEGY_COUNT; ++strategy) {
            arc_program_print_aggregate(&program_aggregates[budget_index][strategy]);
        }
    }
    free(candidates);
    free(evals);
    free(learn_candidates);
    free(learn_evals);
    arc_proposal_trace_prior_free(&proposal_trace_prior);
    if (config->save_learned_state_path != NULL) {
        arc_program_learned_save_state(&learned_state, config->save_learned_state_path);
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
          "[--program-budget-grid] [--program-budget-tight-grid] "
          "[--program-train-dir path] "
          "[--program-train-max-tasks count] [--program-learning-rate rate] "
          "[--hypothesis-library-prior path] [--arc-generative-prior path] "
          "[--hypothesis-prior-hidden count] "
          "[--task-ids comma-list] [--exclude-task-ids comma-list] "
          "[--arc-sketch-probe-candidates path] [--public-arc-dir path] "
          "[--arc-sketch-probe-sources path] [--arc-sketch-probe-prior path] "
          "[--arc-sketch-probe-attempts path] "
          "[--arc-proposal-trace path] "
          "[--arc-holo-residual-diagnostics] [--arc-holo-pressure-residual] "
          "[--arc-holo-bucket-min-budget count] "
          "[--arc-holo-bucket-review-slots count] "
          "[--arc-holo-bucket-min-pressure-updates count] "
          "[--arc-holo-bucket-learned-gate] "
          "[--arc-holo-bucket-learned-gate-min-examples count] "
          "[--arc-holo-bucket-learned-gate-threshold rate] "
          "[--arc-holo-bucket-review-selector-shadow path] "
          "[--arc-sketches-per-task count] "
          "[--hypothesis-top-k count] [--disable-hypothesis-neural-prior] "
          "[--disable-hypothesis-slot-prior] [--disable-hypothesis-cold-start] "
          "[--disable-hypothesis-structural-score] "
          "[--disable-hypothesis-coverage-selection] "
          "[--disable-hypothesis-train-filter] "
          "[--disable-hypothesis-train-consistency-prior] "
          "[--arc-visual-only] [--disable-reflect-tile] [--disable-marker-motif-tile] "
          "[--disable-axis-tile] [--disable-period-extend] "
          "[--arc-runtime-sketch-mine-cap count] "
          "[--arc-generative-prior-op-cap count] "
          "[--save-learned-state path] [--load-learned-state path] [--holo-prior path] [--task-epochs count] [--shuffle] [--unique-task-gate] [--vsa-ranker path] [--vsa-representation grid|object-scene] [--save-hypothesis-prior path] [--hypothesis-library-prior path] [--loader-summary-only] [--induction] "
          "[--min-induction-evidence count] "
          "[--holo-prior path]\n",
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
    config->hypothesis_prior_hidden = ARC_HYPOTHESIS_PRIOR_DEFAULT_HIDDEN;
    config->hypothesis_top_k = ARC_HYPOTHESIS_TOP_K;
    config->arc_runtime_sketch_mine_cap = 16;
    config->arc_generative_prior_op_cap = 2;
    config->seed = 20240516u;
    config->learning_rate = (nn_real)0.006;
    config->program_learning_rate = (nn_real)0.015;
    config->save_path = NULL;
    config->task_dir = NULL;
    config->program_train_dir = NULL;
    config->hypothesis_library_prior_path = NULL;
    config->save_hypothesis_prior_path = NULL;
    config->vsa_ranker_path = NULL;
    config->vsa_representation = ARC_VSA_REPRESENTATION_GRID;
    config->arc_generative_prior_path = NULL;
    config->task_ids = NULL;
    config->exclude_task_ids = NULL;
    config->arc_sketch_probe_candidates_path = NULL;
    config->arc_sketch_probe_public_arc_dir = NULL;
    config->arc_sketch_probe_sources_path = NULL;
    config->arc_sketch_probe_prior_path = NULL;
    config->arc_sketch_probe_attempts_path = NULL;
    config->arc_proposal_trace_path = NULL;
    config->loader_summary_only = 0;
    config->arc_holo_residual_diagnostics = 0;
    config->arc_holo_pressure_residual = 0;
    config->arc_holo_bucket_min_budget = 2;
    config->arc_holo_bucket_review_slots = 0;
    config->arc_holo_bucket_min_pressure_updates = 4;
    config->arc_holo_bucket_learned_gate = 0;
    config->arc_holo_bucket_learned_gate_min_examples = 32;
    config->arc_holo_bucket_learned_gate_threshold = (nn_real)0.50;
    config->arc_holo_bucket_review_selector_shadow_path = NULL;
    config->shuffle_tasks = 0;
    config->unique_task_gate = 0;
    config->program_budget_grid = 0;
    config->induction_smoke = 0;
    config->arc_sketch_probe_sketches_per_task = 8;
    config->hypothesis_neural_prior_enabled = 1;
    config->hypothesis_slot_prior_enabled = 1;
    config->hypothesis_cold_start_enabled = 1;
    config->hypothesis_structural_score_enabled = 1;
    config->hypothesis_coverage_selection_enabled = 1;
    config->hypothesis_train_filter_enabled = 1;
    config->hypothesis_train_consistency_prior_enabled = 1;
    config->arc_visual_only = 0;
    config->disable_reflect_tile = 0;
    config->disable_marker_motif_tile = 0;
    config->disable_axis_tile = 0;
    config->disable_period_extend = 0;
    config->holo_prior_path = NULL;
    config->save_learned_state_path = NULL;
    config->load_learned_state_path = NULL;
    config->task_epochs = 1;
    *schema_only = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--schema") == 0) {
            *schema_only = 1;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--steps-per-epoch") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->steps_per_epoch)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--test-samples") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->test_samples)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
            if (!nn_parse_real_positive_text(argv[++i], &config->learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            if (!nn_parse_u64_text(argv[++i], &config->seed)) {
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
            if (!nn_parse_size_text(argv[++i], &config->max_tasks)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--program-budget") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->program_budget)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--program-budget-grid") == 0) {
            config->program_budget_grid = 1;
        } else if (strcmp(argv[i], "--program-budget-tight-grid") == 0) {
            config->program_budget_grid = 2;
        } else if (strcmp(argv[i], "--program-train-dir") == 0 && i + 1 < argc) {
            config->program_train_dir = argv[++i];
            if (config->program_train_dir[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--program-train-max-tasks") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->program_train_max_tasks)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--program-learning-rate") == 0 && i + 1 < argc) {
            if (!nn_parse_real_positive_text(argv[++i], &config->program_learning_rate)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--holo-prior") == 0 && i + 1 < argc) { config->holo_prior_path = argv[++i]; if (config->holo_prior_path[0] == '\0') return 0;
        } else if (strcmp(argv[i], "--task-epochs") == 0 && i + 1 < argc) { if (!nn_parse_size_text(argv[++i], &config->task_epochs)) return 0;
        } else if (strcmp(argv[i], "--save-hypothesis-prior") == 0 && i + 1 < argc) { config->save_hypothesis_prior_path = argv[++i]; if (config->save_hypothesis_prior_path[0] == '\0') return 0;
        } else if (strcmp(argv[i], "--hypothesis-library-prior") == 0 && i + 1 < argc) {
            config->hypothesis_library_prior_path = argv[++i];
            if (config->hypothesis_library_prior_path[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-generative-prior") == 0 && i + 1 < argc) {
            config->arc_generative_prior_path = argv[++i];
            if (config->arc_generative_prior_path[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--task-ids") == 0 && i + 1 < argc) {
            config->task_ids = argv[++i];
            if (config->task_ids[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--exclude-task-ids") == 0 && i + 1 < argc) {
            config->exclude_task_ids = argv[++i];
            if (config->exclude_task_ids[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-sketch-probe-candidates") == 0 &&
                   i + 1 < argc) {
            config->arc_sketch_probe_candidates_path = argv[++i];
            if (config->arc_sketch_probe_candidates_path[0] == '\0') {
                return 0;
            }
        } else if ((strcmp(argv[i], "--public-arc-dir") == 0 ||
                    strcmp(argv[i], "--arc-sketch-probe-public-arc-dir") == 0) &&
                   i + 1 < argc) {
            config->arc_sketch_probe_public_arc_dir = argv[++i];
            if (config->arc_sketch_probe_public_arc_dir[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-sketch-probe-sources") == 0 &&
                   i + 1 < argc) {
            config->arc_sketch_probe_sources_path = argv[++i];
            if (config->arc_sketch_probe_sources_path[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-sketch-probe-prior") == 0 &&
                   i + 1 < argc) {
            config->arc_sketch_probe_prior_path = argv[++i];
            if (config->arc_sketch_probe_prior_path[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-sketch-probe-attempts") == 0 &&
                   i + 1 < argc) {
            config->arc_sketch_probe_attempts_path = argv[++i];
            if (config->arc_sketch_probe_attempts_path[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-proposal-trace") == 0 &&
                   i + 1 < argc) {
            config->arc_proposal_trace_path = argv[++i];
            if (config->arc_proposal_trace_path[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-holo-residual-diagnostics") == 0) {
            config->arc_holo_residual_diagnostics = 1;
        } else if (strcmp(argv[i], "--arc-holo-pressure-residual") == 0) {
            config->arc_holo_pressure_residual = 1;
        } else if (strcmp(argv[i], "--arc-holo-bucket-min-budget") == 0 &&
                   i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->arc_holo_bucket_min_budget)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-holo-bucket-review-slots") == 0 &&
                   i + 1 < argc) {
            const char *value = argv[++i];
            if (strcmp(value, "0") == 0) {
                config->arc_holo_bucket_review_slots = 0;
            } else if (!nn_parse_size_text(value,
                                           &config->arc_holo_bucket_review_slots)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-holo-bucket-min-pressure-updates") == 0 &&
                   i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i],
                                    &config->arc_holo_bucket_min_pressure_updates)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-holo-bucket-learned-gate") == 0) {
            config->arc_holo_bucket_learned_gate = 1;
        } else if (strcmp(argv[i], "--arc-holo-bucket-learned-gate-min-examples") == 0 &&
                   i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i],
                                    &config->arc_holo_bucket_learned_gate_min_examples)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-holo-bucket-learned-gate-threshold") == 0 &&
                   i + 1 < argc) {
            nn_real value = 0;
            if (!nn_parse_real_nonnegative_text(argv[++i], &value) ||
                value > (nn_real)1) {
                return 0;
            }
            config->arc_holo_bucket_learned_gate_threshold = value;
        } else if (strcmp(argv[i], "--arc-holo-bucket-review-selector-shadow") == 0 &&
                   i + 1 < argc) {
            config->arc_holo_bucket_review_selector_shadow_path = argv[++i];
            if (config->arc_holo_bucket_review_selector_shadow_path[0] == '\0') {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-sketches-per-task") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->arc_sketch_probe_sketches_per_task)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--hypothesis-prior-hidden") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            if (strcmp(value, "0") == 0) {
                config->hypothesis_prior_hidden = 0;
            } else if (!nn_parse_size_text(value, &config->hypothesis_prior_hidden)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--hypothesis-top-k") == 0 && i + 1 < argc) {
            if (!nn_parse_size_text(argv[++i], &config->hypothesis_top_k)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-runtime-sketch-mine-cap") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            if (strcmp(value, "0") == 0) {
                config->arc_runtime_sketch_mine_cap = 0;
            } else if (!nn_parse_size_text(value, &config->arc_runtime_sketch_mine_cap)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--arc-generative-prior-op-cap") == 0 &&
                   i + 1 < argc) {
            const char *value = argv[++i];
            if (strcmp(value, "0") == 0) {
                config->arc_generative_prior_op_cap = 0;
            } else if (!nn_parse_size_text(value,
                                           &config->arc_generative_prior_op_cap)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--disable-hypothesis-neural-prior") == 0) {
            config->hypothesis_neural_prior_enabled = 0;
        } else if (strcmp(argv[i], "--disable-hypothesis-slot-prior") == 0) {
            config->hypothesis_slot_prior_enabled = 0;
        } else if (strcmp(argv[i], "--disable-hypothesis-cold-start") == 0) {
            config->hypothesis_cold_start_enabled = 0;
        } else if (strcmp(argv[i], "--disable-hypothesis-structural-score") == 0) {
            config->hypothesis_structural_score_enabled = 0;
        } else if (strcmp(argv[i], "--disable-hypothesis-coverage-selection") == 0) {
            config->hypothesis_coverage_selection_enabled = 0;
        } else if (strcmp(argv[i], "--disable-hypothesis-train-filter") == 0) {
            config->hypothesis_train_filter_enabled = 0;
        } else if (strcmp(argv[i], "--disable-hypothesis-train-consistency-prior") == 0) {
            config->hypothesis_train_consistency_prior_enabled = 0;
        } else if (strcmp(argv[i], "--arc-visual-only") == 0) {
            config->arc_visual_only = 1;
        } else if (strcmp(argv[i], "--disable-reflect-tile") == 0) {
            config->disable_reflect_tile = 1;
        } else if (strcmp(argv[i], "--disable-marker-motif-tile") == 0) {
            config->disable_marker_motif_tile = 1;
        } else if (strcmp(argv[i], "--disable-axis-tile") == 0) {
            config->disable_axis_tile = 1;
        } else if (strcmp(argv[i], "--disable-period-extend") == 0) {
            config->disable_period_extend = 1;
        } else if (strcmp(argv[i], "--save-learned-state") == 0 && i + 1 < argc) {
            config->save_learned_state_path = argv[++i];
        } else if (strcmp(argv[i], "--load-learned-state") == 0 && i + 1 < argc) {
            config->load_learned_state_path = argv[++i];
        } else if (strcmp(argv[i], "--holo-prior") == 0 && i + 1 < argc) {
            config->holo_prior_path = argv[++i];
        } else if (strcmp(argv[i], "--vsa-ranker") == 0 && i + 1 < argc) { config->vsa_ranker_path = argv[++i]; if (config->vsa_ranker_path[0] == '\0') return 0;
        } else if (strcmp(argv[i], "--vsa-representation") == 0 && i + 1 < argc) { const char *value = argv[++i]; if (strcmp(value, "grid") == 0 || strcmp(value, "arc_grid_hologram_v1") == 0) { config->vsa_representation = ARC_VSA_REPRESENTATION_GRID; } else if (strcmp(value, "object-scene") == 0 || strcmp(value, "arc_object_scene_v1") == 0) { config->vsa_representation = ARC_VSA_REPRESENTATION_OBJECT_SCENE; } else { return 0; }
        } else if (strcmp(argv[i], "--shuffle") == 0) { config->shuffle_tasks = 1;
        } else if (strcmp(argv[i], "--unique-task-gate") == 0) { config->unique_task_gate = 1;
        } else if (strcmp(argv[i], "--loader-summary-only") == 0) {
            config->loader_summary_only = 1;
        } else if (strcmp(argv[i], "--induction") == 0) {
            config->induction_smoke = 1;
        } else if (strcmp(argv[i], "--min-induction-evidence") == 0 && i + 1 < argc) {
            const char *value = argv[++i];
            if (strcmp(value, "0") == 0) {
                config->min_induction_evidence = 0;
            } else if (!nn_parse_size_text(value, &config->min_induction_evidence)) {
                return 0;
            }
        } else {
            return 0;
        }
    }
    return 1;
}

/* Scale-space: downsample input and try to match output dimensions */
static void arc_downsample_max(const ARCGrid *src, ARCGrid *dst, size_t factor) {
    if (factor < 2) { *dst = *src; return; }
    dst->width = src->width / factor; dst->height = src->height / factor;
    for (size_t y = 0; y < dst->height; ++y)
        for (size_t x = 0; x < dst->width; ++x) {
            uint8_t best = 0;
            for (size_t dy = 0; dy < factor; ++dy)
                for (size_t dx = 0; dx < factor; ++dx) {
                    uint8_t v = src->cells[(y * factor + dy) * src->width + (x * factor + dx)];
                    if (v > best) best = v;
                }
            dst->cells[y * dst->width + x] = best;
        }
}

static void arc_upsample_nn(const ARCGrid *src, ARCGrid *dst, size_t factor) {
    if (factor < 2) { *dst = *src; return; }
    dst->width = src->width * factor; dst->height = src->height * factor;
    for (size_t y = 0; y < dst->height; ++y)
        for (size_t x = 0; x < dst->width; ++x)
            dst->cells[y * dst->width + x] = src->cells[(y / factor) * src->width + (x / factor)];
}

static int arc_apply_multiscale_candidate(const ARCTransformCandidate *c, const ARCGrid *in, ARCGrid *out) {
    (void)c;
    if (out->width > ARC_MAX_GRID_WIDTH || out->height > ARC_MAX_GRID_HEIGHT ||
        out->width * out->height > ARC_MAX_GRID_CELLS) return 0;
    size_t best_f = 1;
    for (size_t f = 2; f <= 4; ++f)
        if (in->width % f == 0 && in->height % f == 0) best_f = f;
    ARCGrid small;
    arc_downsample_max(in, &small, best_f);
    if (small.width == out->width && small.height == out->height) { *out = small; return 1; }
    if (out->width % small.width == 0 && out->height % small.height == 0) {
        arc_upsample_nn(&small, out, out->width / small.width);
        return 1;
    }
    arc_init_grid(out, out->width, out->height);
    for (size_t y = 0; y < out->height && y < small.height; ++y)
        for (size_t x = 0; x < out->width && x < small.width; ++x)
            out->cells[y * out->width + x] = small.cells[y * small.width + x];
    return 1;
}

/* Connected component labeling: recolor each component with distinct colors */
static int arc_apply_object_label_candidate(const ARCTransformCandidate *c, const ARCGrid *in, ARCGrid *out) {
    uint8_t visited[ARC_MAX_GRID_CELLS]; memset(visited,0,sizeof(visited));
    arc_init_grid(out,in->width,in->height);
    uint8_t next=1;
    for(size_t y=0;y<in->height;++y)for(size_t x=0;x<in->width;++x){
        size_t idx=y*in->width+x;
        if(visited[idx]||in->cells[idx]==c->background_color)continue;
        static size_t q[ARC_MAX_GRID_CELLS]; size_t qh=0,qt=0;
        q[qt++]=idx; visited[idx]=1;
        while(qh<qt){
            size_t ci=q[qh++]; out->cells[ci]=next;
            size_t cx=ci%in->width,cy=ci/in->width;
            if(cx>0){size_t n=cy*in->width+cx-1;if(!visited[n]&&in->cells[n]!=c->background_color){visited[n]=1;q[qt++]=n;}}
            if(cx+1<in->width){size_t n=cy*in->width+cx+1;if(!visited[n]&&in->cells[n]!=c->background_color){visited[n]=1;q[qt++]=n;}}
            if(cy>0){size_t n=(cy-1)*in->width+cx;if(!visited[n]&&in->cells[n]!=c->background_color){visited[n]=1;q[qt++]=n;}}
            if(cy+1<in->height){size_t n=(cy+1)*in->width+cx;if(!visited[n]&&in->cells[n]!=c->background_color){visited[n]=1;q[qt++]=n;}}
        }
        if(next<9)++next;
    }
    return 1;
}
static int ARC_NOINLINE run_builtin_arc_experiment(const ARCConfig *config)
{
    if (config == NULL) {
        return EXIT_FAILURE;
    }
    NeuralNet *net = create_arc_model((unsigned int)(config->seed & 0xffffffffu));
    if (net == NULL) {
        return EXIT_FAILURE;
    }
    if (!train_arc_model(net, config)) {
        nn_free(net);
        return EXIT_FAILURE;
    }

    char model_hash[65] = "unknown";
    if (config->save_path != NULL) {
        if (!save_arc_checkpoint(net, config->save_path, model_hash)) {
            fprintf(stderr, "failed to save ARC checkpoint: %s\n", config->save_path);
            nn_free(net);
            return EXIT_FAILURE;
        }
        printf("saved arc_agi checkpoint: %s hash=%s\n", config->save_path, model_hash);
    }

    ARCMetrics metrics = evaluate_arc_model(net, config->test_samples);
    print_arc_metrics_json("0520fde7", model_hash, "neural_cell_scorer", "arc-cell-color-v1", &metrics);
    print_arc_metrics_line(model_hash, "neural_cell_scorer", &metrics);

    ARCTask builtin_task;
    init_builtin_arc_task(&builtin_task);
    ARCTransformMetrics transform = evaluate_builtin_arc_transform_gate();
    print_arc_transform_json("0520fde7", &transform);
    ARCMetrics transform_metrics = arc_transform_to_metrics(&builtin_task, &transform);
    print_arc_metrics_json("0520fde7",
                          "symbolic-transform-v14",
                           "certified_transform_gate",
                          "arc-transform-gate-v14",
                           &transform_metrics);
    print_arc_metrics_line("symbolic-transform-v14",
                           "certified_transform_gate",
                           &transform_metrics);
    arc_emit_builtin_program_summaries(&builtin_task, config);
    nn_free(net);
    return EXIT_SUCCESS;
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
    if (config.arc_sketch_probe_candidates_path != NULL ||
        config.arc_sketch_probe_public_arc_dir != NULL ||
        config.arc_sketch_probe_sources_path != NULL ||
        config.arc_sketch_probe_prior_path != NULL ||
        config.arc_sketch_probe_attempts_path != NULL) {
        return arc_export_sketch_program_probes(&config) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (config.task_dir != NULL) {
        return run_arc_task_loader(&config) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    return run_builtin_arc_experiment(&config);
}
/* Holographic tile detector: detect cyclic pattern period and tile */
static int arc_apply_holo_tile_candidate(const ARCTransformCandidate *c, const ARCGrid *in, ARCGrid *out) {
    if (out->width > ARC_MAX_GRID_WIDTH || out->height > ARC_MAX_GRID_HEIGHT || out->width * out->height > ARC_MAX_GRID_CELLS) return 0;
    static size_t xs[900],ys[900]; static uint8_t cs[900]; size_t n=0;
    for(size_t y=0;y<in->height&&n<900;++y)for(size_t x=0;x<in->width&&n<900;++x){
        uint8_t v=in->cells[y*in->width+x]; if(v!=c->background_color){xs[n]=x;ys[n]=y;cs[n]=v;++n;}
    }
    if(n<2){arc_init_grid(out,in->width,in->height);for(size_t i=0;i<arc_grid_cells(in);++i)out->cells[i]=in->cells[i];return 1;}

    /* Fast heuristic first: detect period from same-color spacing */
    size_t period=in->width;
    for(size_t i=0;i<n;++i)for(size_t j=i+1;j<n;++j)
        if(cs[i]==cs[j]){size_t d=xs[j]>xs[i]?xs[j]-xs[i]:xs[i]-xs[j];if(d>0&&d<period)period=d;}
    if(period<2||period>30)period=in->width;

    /* VSA verification skipped: fast heuristic is sufficient */

    static uint8_t row[30]; memset(row,0,sizeof(row));
    size_t by=0;int fd=0;
    for(size_t y=0;y<in->height&&!fd;++y)for(size_t x=0;x<in->width;++x)
        if(in->cells[y*in->width+x]!=c->background_color){by=y;fd=1;break;}
    for(size_t x=0;x<in->width;++x){uint8_t v=in->cells[by*in->width+x];if(v!=c->background_color)row[x%period]=v;}
    for(size_t y=0;y<in->height;++y)for(size_t x=0;x<in->width;++x){
        uint8_t v=in->cells[y*in->width+x];if(v!=c->background_color){size_t px=x%period;if(!row[px])row[px]=v;}
    }
    arc_init_grid(out,out->width,out->height);
    for(size_t y=0;y<out->height;++y){size_t sh=y%period;for(size_t x=0;x<out->width;++x)out->cells[y*out->width+x]=row[(x+sh)%period];}
    return 1;
}
/* Reusable holographic grid encoder: grid → 256-dim scene vector.
 * Precomputes position keys (900 cells) and color keys (10 colors) once.
 * Caller provides pre-allocated double[256]. Returns 0 on success. */
static int arc_grid_to_hologram(const ARCGrid *grid, double *scene_out) {
    static LeosEngine *eng = NULL;
    static double pos_keys[900][256];
    static double col_keys[10][256];
    static int init = 0;
    if (!init) {
        eng = leos_engine_create(256);
        if (!eng) return -1;
        for (size_t y = 0; y < 30; ++y)
            for (size_t x = 0; x < 30; ++x)
                leos_engine_keygen(eng, (uint64_t)(y * 1000 + x), pos_keys[y * 30 + x]);
        for (uint8_t c = 0; c < 10; ++c)
            leos_engine_keygen(eng, (uint64_t)(c * 10000 + 1), col_keys[c]);
        init = 1;
    }
    if (!eng) return -1;
    memset(scene_out, 0, 256 * sizeof(double));
    double cell[256]; size_t count = 0;
    for (size_t y = 0; y < grid->height; ++y)
        for (size_t x = 0; x < grid->width; ++x) {
            uint8_t v = grid->cells[y * grid->width + x];
            if (v == 0 || v >= 10) continue;
            if (++count > 200) break;
            leos_engine_bind(eng, pos_keys[y * 30 + x], col_keys[v], cell);
            for (size_t j = 0; j < 256; ++j) scene_out[j] += cell[j];
        }
    leos_engine_normalize(eng, scene_out);
    return 0;
}

static size_t arc_vsa_bucket_size(size_t value)
{
    if (value == 0) return 0;
    if (value <= 1) return 1;
    if (value <= 2) return 2;
    if (value <= 4) return 3;
    if (value <= 8) return 4;
    if (value <= 16) return 5;
    if (value <= 32) return 6;
    if (value <= 64) return 7;
    if (value <= 128) return 8;
    return 9;
}

static size_t arc_vsa_bucket_coord(double value, size_t limit)
{
    if (limit <= 1) return 0;
    double scaled = value * 7.0 / (double)(limit - 1u);
    if (scaled < 0.0) return 0;
    if (scaled > 7.0) return 7;
    return (size_t)(scaled + 0.5);
}

static void arc_vsa_add_feature(LeosEngine *eng,
                                double *out,
                                uint64_t attr_seed,
                                uint64_t value_seed,
                                double *attr_vec,
                                double *value_vec,
                                double *bound)
{
    leos_engine_keygen(eng, attr_seed, attr_vec);
    leos_engine_keygen(eng, value_seed, value_vec);
    leos_engine_bind(eng, attr_vec, value_vec, bound);
    for (size_t j = 0; j < 256; ++j) out[j] += bound[j];
}

static void arc_vsa_add_object_feature(LeosEngine *eng,
                                       double *object_vec,
                                       uint64_t attr,
                                       uint64_t value,
                                       double *attr_vec,
                                       double *value_vec,
                                       double *bound)
{
    arc_vsa_add_feature(eng,
                        object_vec,
                        UINT64_C(0x9a1d000000000000) ^ attr,
                        UINT64_C(0x9a1e000000000000) ^ value,
                        attr_vec,
                        value_vec,
                        bound);
}

static int arc_object_scene_to_hologram(const ARCGrid *grid, double *scene_out)
{
    static LeosEngine *eng = NULL;
    if (eng == NULL) {
        eng = leos_engine_create(256);
        if (eng == NULL) return -1;
    }
    ARCScene scene;
    int complete_scene = arc_extract_scene(grid, 0, &scene);
    memset(scene_out, 0, 256 * sizeof(double));
    double attr_vec[256], value_vec[256], bound[256], object_vec[256], slot_vec[256];
    size_t color_objects[10] = {0};
    size_t foreground_cells = 0;
    for (size_t i = 0; i < scene.object_count; ++i) {
        const ARCObject *object = &scene.objects[i];
        if (object->color < 10) ++color_objects[object->color];
        foreground_cells += object->count;
    }

    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000001), UINT64_C(0x9b02000000000000) ^ grid->width, attr_vec, value_vec, bound);
    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000002), UINT64_C(0x9b02000000000020) ^ grid->height, attr_vec, value_vec, bound);
    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000003), UINT64_C(0x9b02000000000040) ^ arc_vsa_bucket_size(scene.object_count), attr_vec, value_vec, bound);
    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000004), UINT64_C(0x9b02000000000060) ^ arc_vsa_bucket_size(foreground_cells), attr_vec, value_vec, bound);
    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000005), UINT64_C(0x9b02000000000080) ^ arc_vsa_bucket_size(scene.repeated_shape_pairs), attr_vec, value_vec, bound);
    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000006), UINT64_C(0x9b020000000000a0) ^ arc_vsa_bucket_size(scene.aligned_pairs), attr_vec, value_vec, bound);
    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000007), UINT64_C(0x9b020000000000c0) ^ arc_vsa_bucket_size(scene.adjacent_pairs), attr_vec, value_vec, bound);
    arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b01000000000008), UINT64_C(0x9b020000000000e0) ^ (complete_scene ? 1u : 0u), attr_vec, value_vec, bound);
    for (uint8_t color = 1; color < 10; ++color) {
        arc_vsa_add_feature(eng, scene_out, UINT64_C(0x9b03000000000000) ^ color, UINT64_C(0x9b04000000000000) ^ arc_vsa_bucket_size(color_objects[color]), attr_vec, value_vec, bound);
    }

    for (size_t i = 0; i < scene.object_count; ++i) {
        const ARCObject *object = &scene.objects[i];
        size_t width = object->max_x - object->min_x + 1u;
        size_t height = object->max_y - object->min_y + 1u;
        memset(object_vec, 0, sizeof(object_vec));
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x01), UINT64_C(0x100) ^ object->color, attr_vec, value_vec, bound);
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x02), UINT64_C(0x200) ^ arc_vsa_bucket_size(object->count), attr_vec, value_vec, bound);
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x03), UINT64_C(0x300) ^ arc_vsa_bucket_size(width), attr_vec, value_vec, bound);
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x04), UINT64_C(0x400) ^ arc_vsa_bucket_size(height), attr_vec, value_vec, bound);
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x05), UINT64_C(0x500) ^ arc_vsa_bucket_coord(object->centroid_x, grid->width), attr_vec, value_vec, bound);
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x06), UINT64_C(0x600) ^ arc_vsa_bucket_coord(object->centroid_y, grid->height), attr_vec, value_vec, bound);
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x07), UINT64_C(0x700) ^ (object->touches_border ? 1u : 0u), attr_vec, value_vec, bound);
        arc_vsa_add_object_feature(eng, object_vec, UINT64_C(0x08), UINT64_C(0x800) ^ (object->shape_hash & UINT64_C(0xffff)), attr_vec, value_vec, bound);
        leos_engine_normalize(eng, object_vec);
        leos_engine_keygen(eng, UINT64_C(0x9b05000000000000) ^ i, slot_vec);
        leos_engine_bind(eng, slot_vec, object_vec, bound);
        for (size_t j = 0; j < 256; ++j) scene_out[j] += bound[j];
    }
    leos_engine_normalize(eng, scene_out);
    return 0;
}
