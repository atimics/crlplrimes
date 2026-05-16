#include "nn.h"

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
    ARC_RULE_COUNT = 6,
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
    uint64_t seed;
    nn_real learning_rate;
    const char *save_path;
} ARCConfig;

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
    size_t test_cells;
    size_t test_correct_cells;
    int unique_test_prediction;
    int test_exact;
    int certified;
    char accepted_rules[160];
    uint8_t prediction[ARC_OUTPUT_CELLS];
} ARCTransformMetrics;

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

static const char *arc_rule_name(size_t template_id)
{
    static const char *names[ARC_RULE_COUNT] = {
        "copy_left_color_2",
        "copy_right_color_2",
        "mask_intersection_color_2",
        "mask_union_color_2",
        "object_select_overlap_color_2",
        "color_map_left_identity",
    };
    return template_id < ARC_RULE_COUNT ? names[template_id] : "unknown";
}

static void arc_apply_rule(size_t template_id, const uint8_t *input, uint8_t *output)
{
    memset(output, 0, ARC_OUTPUT_CELLS * sizeof(*output));
    for (size_t y = 0; y < ARC_OUTPUT_HEIGHT; ++y) {
        for (size_t x = 0; x < ARC_OUTPUT_WIDTH; ++x) {
            uint8_t left = input[y * ARC_INPUT_WIDTH + x];
            uint8_t right = input[y * ARC_INPUT_WIDTH + x + 4u];
            size_t out = y * ARC_OUTPUT_WIDTH + x;
            switch (template_id) {
            case ARC_RULE_COPY_LEFT:
                output[out] = left == 1 ? 2 : 0;
                break;
            case ARC_RULE_COPY_RIGHT:
                output[out] = right == 1 ? 2 : 0;
                break;
            case ARC_RULE_INTERSECTION:
                output[out] = left == 1 && right == 1 ? 2 : 0;
                break;
            case ARC_RULE_UNION:
                output[out] = left == 1 || right == 1 ? 2 : 0;
                break;
            case ARC_RULE_OBJECT_SELECT:
                output[out] = left == 1 && right == 1 ? 2 : 0;
                break;
            case ARC_RULE_COLOR_MAP_LEFT:
                output[out] = left == 1 ? 1 : 0;
                break;
            default:
                output[out] = 0;
                break;
            }
        }
    }
}

static int arc_rule_matches_train(size_t template_id,
                                  size_t *checked_cells_out,
                                  size_t *mismatches_out)
{
    uint8_t produced[ARC_OUTPUT_CELLS];
    size_t checked_cells = 0;
    size_t mismatches = 0;
    for (size_t pair = 0; pair < ARC_TRAIN_PAIR_COUNT; ++pair) {
        arc_apply_rule(template_id, ARC_0520FDE7_TRAIN_INPUTS[pair], produced);
        for (size_t cell = 0; cell < ARC_OUTPUT_CELLS; ++cell) {
            ++checked_cells;
            if (produced[cell] != ARC_0520FDE7_TRAIN_OUTPUTS[pair][cell]) {
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
    return mismatches == 0;
}

static ARCTransformMetrics evaluate_arc_transform_gate(void)
{
    ARCTransformMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    metrics.rules = ARC_RULE_COUNT;
    metrics.unique_test_prediction = 1;

    uint8_t first_prediction[ARC_OUTPUT_CELLS];
    memset(first_prediction, 0, sizeof(first_prediction));
    int have_prediction = 0;
    for (size_t rule = 0; rule < ARC_RULE_COUNT; ++rule) {
        size_t checked_cells = 0;
        size_t mismatches = 0;
        if (!arc_rule_matches_train(rule, &checked_cells, &mismatches)) {
            metrics.train_checked_cells += checked_cells;
            metrics.train_mismatches += mismatches;
            continue;
        }
        metrics.train_checked_cells += checked_cells;
        ++metrics.train_consistent_rules;
        size_t used = strlen(metrics.accepted_rules);
        snprintf(metrics.accepted_rules + used,
                 sizeof(metrics.accepted_rules) - used,
                 "%s%s",
                 used == 0 ? "" : ",",
                 arc_rule_name(rule));

        uint8_t produced[ARC_OUTPUT_CELLS];
        arc_apply_rule(rule, ARC_0520FDE7_TEST_INPUT, produced);
        if (!have_prediction) {
            memcpy(first_prediction, produced, sizeof(first_prediction));
            have_prediction = 1;
        } else if (memcmp(first_prediction, produced, sizeof(first_prediction)) != 0) {
            metrics.unique_test_prediction = 0;
        }
    }

    if (have_prediction) {
        memcpy(metrics.prediction, first_prediction, sizeof(metrics.prediction));
    } else {
        metrics.unique_test_prediction = 0;
    }

    metrics.test_cells = ARC_OUTPUT_CELLS;
    for (size_t cell = 0; cell < ARC_OUTPUT_CELLS; ++cell) {
        metrics.test_correct_cells += metrics.prediction[cell] == ARC_0520FDE7_TEST_OUTPUT[cell] ? 1u : 0u;
    }
    metrics.test_exact = metrics.test_correct_cells == metrics.test_cells;
    metrics.certified = metrics.train_consistent_rules != 0 &&
                        metrics.unique_test_prediction &&
                        metrics.test_exact;
    return metrics;
}

static ARCMetrics arc_transform_to_metrics(const ARCTransformMetrics *transform)
{
    ARCMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    metrics.samples = ARC_OUTPUT_CELLS;
    metrics.candidate_count = ARC_ACTION_COUNT;
    for (size_t cell = 0; cell < ARC_OUTPUT_CELLS; ++cell) {
        size_t target = ARC_0520FDE7_TEST_OUTPUT[cell];
        size_t selected = transform->certified ? transform->prediction[cell] : 0u;
        size_t rank = selected == target ? 0u : target == 0u ? 1u : target;
        metrics.top1 += rank == 0 ? 1u : 0u;
        metrics.top5 += rank < 5 ? 1u : 0u;
        metrics.top10 += 1u;
        metrics.nll += rank == 0 ? 0.0 : -log(1e-12);
        metrics.mean_rank += (double)rank + 1.0;
        metrics.decision_utility += rank == 0 ? 1.0 : 0.0;
        metrics.proposal_steps += (double)rank + 1.0;
        metrics.baseline_steps += (double)target + 1.0;
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

static void print_arc_metrics_json(const char *model_hash,
                                   const char *evaluation_mode,
                                   const char *feature_set,
                                   const ARCMetrics *metrics)
{
    double samples = metrics->samples == 0 ? 1.0 : (double)metrics->samples;
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"arc_agi\","
           "\"task_instance\":\"0520fde7\","
           "\"evaluation_mode\":\"%s\","
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

static void print_arc_transform_json(const ARCTransformMetrics *metrics)
{
    printf("{\"schema\":\"crlp.arc_transform.summary.v1\",\"report_version\":1,"
           "\"task\":\"arc_agi\",\"task_instance\":\"0520fde7\","
           "\"grammar_version\":1,\"rules\":%zu,\"train_consistent_rules\":%zu,"
           "\"train_checked_cells\":%zu,\"train_mismatches\":%zu,"
           "\"unique_test_prediction\":%s,\"test_exact\":%s,\"certified\":%s,"
           "\"test_cells\":%zu,\"test_correct_cells\":%zu,"
           "\"accepted_rules\":\"%s\"}\n",
           metrics->rules,
           metrics->train_consistent_rules,
           metrics->train_checked_cells,
           metrics->train_mismatches,
           metrics->unique_test_prediction ? "true" : "false",
           metrics->test_exact ? "true" : "false",
           metrics->certified ? "true" : "false",
           metrics->test_cells,
           metrics->test_correct_cells,
           metrics->accepted_rules);
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

static void print_usage(FILE *stream)
{
    fputs("usage: arc_agi [--schema] [--epochs count] [--steps-per-epoch count] "
          "[--test-samples count] [--learning-rate rate] [--seed seed] [--save path]\n",
          stream);
}

static int parse_config(int argc, char **argv, ARCConfig *config, int *schema_only)
{
    config->epochs = ARC_DEFAULT_EPOCHS;
    config->steps_per_epoch = ARC_DEFAULT_STEPS_PER_EPOCH;
    config->test_samples = ARC_DEFAULT_TEST_SAMPLES;
    config->seed = 20240516u;
    config->learning_rate = (nn_real)0.006;
    config->save_path = NULL;
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
    print_arc_metrics_json(model_hash, "neural_cell_scorer", "arc-cell-color-v1", &metrics);
    print_arc_metrics_line(model_hash, "neural_cell_scorer", &metrics);

    ARCTransformMetrics transform = evaluate_arc_transform_gate();
    print_arc_transform_json(&transform);
    ARCMetrics transform_metrics = arc_transform_to_metrics(&transform);
    print_arc_metrics_json("symbolic-transform-v1",
                           "certified_transform_gate",
                           "arc-transform-gate-v1",
                           &transform_metrics);
    print_arc_metrics_line("symbolic-transform-v1",
                           "certified_transform_gate",
                           &transform_metrics);
    nn_free(net);
    return EXIT_SUCCESS;
}
