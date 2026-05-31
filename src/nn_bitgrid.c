#include "nn_bitgrid.h"
#include "nn_train.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

enum {
    BG_INPUT_WIDTH = 2,
    BG_OUTPUT_WIDTH = 2,
    BG_TRUTH_ROWS = 4,
    BG_TRAIN_CHUNK_EPOCHS = 64,
    BG_CHAIN_INITIAL_NODES = 4,
};

static size_t bg_min_size(size_t a, size_t b)
{
    return a < b ? a : b;
}

static int bg_finite(nn_real value)
{
    return isfinite((double)value);
}

static void bg_report_init(NNBitwiseTrainReport *report, uint8_t truth_mask)
{
    if (report == NULL) {
        return;
    }
    memset(report, 0, sizeof(*report));
    snprintf(report->schema, sizeof(report->schema), "%s", NN_BITGRID_GATE_SCHEMA);
    report->op = (NNBitwiseOp)(truth_mask & 0x0fu);
    report->truth_mask = (uint8_t)(truth_mask & 0x0fu);
    report->arity = 2u;
    report->input_width = BG_INPUT_WIDTH;
    report->output_width = BG_OUTPUT_WIDTH;
    report->sample_count = BG_TRUTH_ROWS;
    report->final_loss = (nn_real)NAN;
    report->min_margin = (nn_real)-INFINITY;
}

static void bg_truth_table(uint8_t truth_mask, uint8_t table[BG_TRUTH_ROWS])
{
    for (size_t i = 0; i < BG_TRUTH_ROWS; ++i) {
        table[i] = (uint8_t)((truth_mask >> i) & 1u);
    }
}

static void bg_build_dataset(uint8_t truth_mask,
                             nn_real inputs[BG_TRUTH_ROWS * BG_INPUT_WIDTH],
                             nn_real targets[BG_TRUTH_ROWS * BG_OUTPUT_WIDTH])
{
    for (size_t i = 0; i < BG_TRUTH_ROWS; ++i) {
        uint8_t a = (uint8_t)((i >> 1u) & 1u);
        uint8_t b = (uint8_t)(i & 1u);
        uint8_t y = (uint8_t)((truth_mask >> i) & 1u);
        inputs[i * BG_INPUT_WIDTH] = (nn_real)a;
        inputs[i * BG_INPUT_WIDTH + 1u] = (nn_real)b;
        targets[i * BG_OUTPUT_WIDTH] = y == 0u ? (nn_real)1 : (nn_real)0;
        targets[i * BG_OUTPUT_WIDTH + 1u] = y == 1u ? (nn_real)1 : (nn_real)0;
    }
}

static int bg_eval_net(NeuralNet *net,
                       uint8_t truth_mask,
                       uint8_t predicted[BG_TRUTH_ROWS],
                       NNBitwiseTrainReport *report)
{
    if (net == NULL || predicted == NULL || report == NULL) {
        return 0;
    }

    nn_real inputs[BG_TRUTH_ROWS * BG_INPUT_WIDTH];
    nn_real targets[BG_TRUTH_ROWS * BG_OUTPUT_WIDTH];
    bg_build_dataset(truth_mask, inputs, targets);

    size_t correct = 0;
    nn_real loss = (nn_real)0;
    nn_real min_margin = (nn_real)INFINITY;
    int finite = 1;

    for (size_t i = 0; i < BG_TRUTH_ROWS; ++i) {
        nn_real logits[BG_OUTPUT_WIDTH];
        nn_real probabilities[BG_OUTPUT_WIDTH];
        nn_forward(net, &inputs[i * BG_INPUT_WIDTH], logits);
        if (!bg_finite(logits[0]) || !bg_finite(logits[1])) {
            probabilities[0] = (nn_real)NAN;
            probabilities[1] = (nn_real)NAN;
            finite = 0;
        } else {
            nn_softmax(logits, probabilities, BG_OUTPUT_WIDTH);
            loss += nn_softmax_cross_entropy(logits,
                                             &targets[i * BG_OUTPUT_WIDTH],
                                             BG_OUTPUT_WIDTH);
        }

        uint8_t expected = (uint8_t)((truth_mask >> i) & 1u);
        uint8_t actual =
            bg_finite(probabilities[0]) && bg_finite(probabilities[1]) &&
                    probabilities[1] >= probabilities[0]
                ? 1u
                : 0u;
        predicted[i] = actual;
        if (actual == expected) {
            ++correct;
        }

        nn_real target_probability = probabilities[expected ? 1u : 0u];
        nn_real other_probability = probabilities[expected ? 0u : 1u];
        nn_real margin = target_probability - other_probability;
        if (!bg_finite(margin)) {
            finite = 0;
            margin = (nn_real)-INFINITY;
        }
        if (margin < min_margin) {
            min_margin = margin;
        }
    }

    report->final_loss = finite ? loss / (nn_real)BG_TRUTH_ROWS : (nn_real)NAN;
    report->min_margin = min_margin;
    report->accuracy = (double)correct / (double)BG_TRUTH_ROWS;
    report->learned_truth_table = correct == BG_TRUTH_ROWS && finite;
    return finite;
}

int nn_bitgrid_init(NNBitGrid *grid, size_t width, size_t height)
{
    if (grid == NULL || width == 0 || height == 0 || width > SIZE_MAX / height) {
        return 0;
    }

    memset(grid, 0, sizeof(*grid));
    size_t cells = width * height;
    grid->cells = (uint8_t *)calloc(cells, sizeof(*grid->cells));
    if (grid->cells == NULL) {
        return 0;
    }
    grid->width = width;
    grid->height = height;
    return 1;
}

void nn_bitgrid_free(NNBitGrid *grid)
{
    if (grid == NULL) {
        return;
    }
    free(grid->cells);
    memset(grid, 0, sizeof(*grid));
}

size_t nn_bitgrid_cell_count(const NNBitGrid *grid)
{
    if (grid == NULL || grid->cells == NULL) {
        return 0;
    }
    return grid->width * grid->height;
}

int nn_bitgrid_same_shape(const NNBitGrid *a, const NNBitGrid *b)
{
    return a != NULL && b != NULL && a->cells != NULL && b->cells != NULL &&
           a->width == b->width && a->height == b->height;
}

int nn_bitgrid_get(const NNBitGrid *grid, size_t x, size_t y, uint8_t *out_bit)
{
    if (grid == NULL || grid->cells == NULL || out_bit == NULL || x >= grid->width ||
        y >= grid->height) {
        return 0;
    }
    *out_bit = (uint8_t)(grid->cells[y * grid->width + x] & 1u);
    return 1;
}

int nn_bitgrid_set(NNBitGrid *grid, size_t x, size_t y, uint8_t bit)
{
    if (grid == NULL || grid->cells == NULL || x >= grid->width || y >= grid->height) {
        return 0;
    }
    grid->cells[y * grid->width + x] = (uint8_t)(bit & 1u);
    return 1;
}

int nn_bitgrid_fill(NNBitGrid *grid, uint8_t bit)
{
    if (grid == NULL || grid->cells == NULL) {
        return 0;
    }
    size_t cells = nn_bitgrid_cell_count(grid);
    for (size_t i = 0; i < cells; ++i) {
        grid->cells[i] = (uint8_t)(bit & 1u);
    }
    return 1;
}

int nn_bitgrid_fill_alternating(NNBitGrid *grid, uint8_t first_bit)
{
    if (grid == NULL || grid->cells == NULL) {
        return 0;
    }
    size_t cells = nn_bitgrid_cell_count(grid);
    for (size_t i = 0; i < cells; ++i) {
        grid->cells[i] = (uint8_t)((first_bit + i) & 1u);
    }
    return 1;
}

int nn_bitgrid_copy(NNBitGrid *dst, const NNBitGrid *src)
{
    if (!nn_bitgrid_same_shape(dst, src)) {
        return 0;
    }
    memcpy(dst->cells, src->cells, nn_bitgrid_cell_count(src) * sizeof(*dst->cells));
    return 1;
}

size_t nn_bitgrid_hamming_distance(const NNBitGrid *a, const NNBitGrid *b)
{
    if (!nn_bitgrid_same_shape(a, b)) {
        return SIZE_MAX;
    }
    size_t distance = 0;
    size_t cells = nn_bitgrid_cell_count(a);
    for (size_t i = 0; i < cells; ++i) {
        distance += (a->cells[i] & 1u) != (b->cells[i] & 1u) ? 1u : 0u;
    }
    return distance;
}

int nn_bitgrid_write_pattern(const NNBitGrid *grid, char *buffer, size_t buffer_size)
{
    if (grid == NULL || grid->cells == NULL || buffer == NULL || buffer_size == 0) {
        return 0;
    }

    size_t separators = grid->height > 0 ? grid->height - 1u : 0u;
    size_t required = nn_bitgrid_cell_count(grid) + separators + 1u;
    if (required > buffer_size) {
        return 0;
    }

    size_t cursor = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        if (y != 0) {
            buffer[cursor++] = '/';
        }
        for (size_t x = 0; x < grid->width; ++x) {
            buffer[cursor++] = grid->cells[y * grid->width + x] ? '1' : '0';
        }
    }
    buffer[cursor] = '\0';
    return 1;
}

const char *nn_bitwise_op_name(NNBitwiseOp op)
{
    return nn_bitwise_truth_mask_name(nn_bitwise_op_truth_mask(op));
}

size_t nn_bitwise_op_arity(NNBitwiseOp op)
{
    (void)op;
    return 2u;
}

uint8_t nn_bitwise_op_truth_mask(NNBitwiseOp op)
{
    return (uint8_t)op & 0x0fu;
}

const char *nn_bitwise_truth_mask_name(uint8_t truth_mask)
{
    switch (truth_mask & 0x0fu) {
    case NN_BITWISE_FALSE:
        return "false";
    case NN_BITWISE_NOR:
        return "nor";
    case NN_BITWISE_NOT_A_AND_B:
        return "not_a_and_b";
    case NN_BITWISE_NOT_A:
        return "not_a";
    case NN_BITWISE_A_AND_NOT_B:
        return "a_and_not_b";
    case NN_BITWISE_NOT_B:
        return "not_b";
    case NN_BITWISE_XOR:
        return "xor";
    case NN_BITWISE_NAND:
        return "nand";
    case NN_BITWISE_AND:
        return "and";
    case NN_BITWISE_XNOR:
        return "xnor";
    case NN_BITWISE_B:
        return "b";
    case NN_BITWISE_NOT_A_OR_B:
        return "not_a_or_b";
    case NN_BITWISE_A:
        return "a";
    case NN_BITWISE_A_OR_NOT_B:
        return "a_or_not_b";
    case NN_BITWISE_OR:
        return "or";
    case NN_BITWISE_TRUE:
        return "true";
    default:
        return "unknown";
    }
}

int nn_bitwise_truth_mask_eval(uint8_t truth_mask, uint8_t a, uint8_t b, uint8_t *out_bit)
{
    if ((truth_mask & 0xf0u) != 0u || out_bit == NULL) {
        return 0;
    }
    uint8_t index = (uint8_t)(((a & 1u) << 1u) | (b & 1u));
    *out_bit = (uint8_t)((truth_mask >> index) & 1u);
    return 1;
}

int nn_bitwise_op_eval(NNBitwiseOp op, uint8_t a, uint8_t b, uint8_t *out_bit)
{
    return nn_bitwise_truth_mask_eval(nn_bitwise_op_truth_mask(op), a, b, out_bit);
}

uint8_t nn_bitwise_apply_truth_mask_to_masks(uint8_t op_mask,
                                             uint8_t left_mask,
                                             uint8_t right_mask)
{
    uint8_t result = 0u;
    for (uint8_t row = 0; row < BG_TRUTH_ROWS; ++row) {
        uint8_t a = (uint8_t)((left_mask >> row) & 1u);
        uint8_t b = (uint8_t)((right_mask >> row) & 1u);
        uint8_t out = 0u;
        if (!nn_bitwise_truth_mask_eval((uint8_t)(op_mask & 0x0fu), a, b, &out)) {
            return 0u;
        }
        result |= (uint8_t)(out << row);
    }
    return result;
}

uint16_t nn_bitwise_default_tool_mask(void)
{
    return (uint16_t)((1u << NN_BITWISE_AND) |
                      (1u << NN_BITWISE_OR) |
                      (1u << NN_BITWISE_XOR) |
                      (1u << NN_BITWISE_NOT_A));
}

void nn_bitwise_train_config_init(NNBitwiseTrainConfig *config)
{
    if (config == NULL) {
        return;
    }
    config->hidden_width = 8u;
    config->max_epochs = 8192u;
    config->batch_size = BG_TRUTH_ROWS;
    config->retry_count = 8u;
    config->learning_rate = (nn_real)0.08;
    config->min_margin = (nn_real)0.20;
    config->seed = 20240522u;
    config->shuffle = 1;
}

void nn_bitwise_gate_init(NNBitwiseGate *gate)
{
    if (gate == NULL) {
        return;
    }
    memset(gate, 0, sizeof(*gate));
    gate->op = NN_BITWISE_FALSE;
    gate->truth_mask = 0u;
    gate->arity = 2u;
    nn_certificate_clear(&gate->certificate);
    bg_report_init(&gate->report, 0u);
}

void nn_bitwise_gate_free(NNBitwiseGate *gate)
{
    if (gate == NULL) {
        return;
    }
    nn_free(gate->net);
    gate->net = NULL;
    nn_certificate_clear(&gate->certificate);
}

static int bg_report_is_better(const NNBitwiseTrainReport *candidate,
                               const NNBitwiseTrainReport *best)
{
    if (best->attempts == 0) {
        return 1;
    }
    if (candidate->accuracy != best->accuracy) {
        return candidate->accuracy > best->accuracy;
    }
    return candidate->min_margin > best->min_margin;
}

void nn_bitwise_chain_init(NNBitwiseChain *chain)
{
    if (chain == NULL) {
        return;
    }
    memset(chain, 0, sizeof(*chain));
    chain->output_index = SIZE_MAX;
}

void nn_bitwise_tool_policy_config_init(NNBitwiseToolPolicyConfig *config)
{
    if (config == NULL) {
        return;
    }
    config->hidden_width = 32u;
    config->epochs = 48u;
    config->learning_rate = (nn_real)0.05;
    config->seed = 20240522u;
}

void nn_bitwise_tool_policy_init(NNBitwiseToolPolicy *policy)
{
    if (policy == NULL) {
        return;
    }
    memset(policy, 0, sizeof(*policy));
    policy->allowed_truth_masks = nn_bitwise_default_tool_mask();
    policy->max_steps = 4u;
    policy->final_loss = (nn_real)NAN;
}

void nn_bitwise_tool_policy_free(NNBitwiseToolPolicy *policy)
{
    if (policy == NULL) {
        return;
    }
    nn_free(policy->net);
    policy->net = NULL;
}

static int bg_tool_is_allowed(uint16_t allowed_truth_masks, uint8_t op_mask)
{
    return op_mask < NN_BITWISE_BINARY_TRUTH_TABLES &&
           ((allowed_truth_masks >> op_mask) & 1u) != 0u;
}

static int bg_chain_seed(NNBitwiseChain *chain)
{
    if (chain == NULL) {
        return 0;
    }
    nn_bitwise_chain_init(chain);
    chain->nodes[0] = (NNBitwiseChainNode){
        .kind = NN_BITWISE_CHAIN_INPUT_X,
        .truth_mask = NN_BITWISE_A,
        .op_mask = NN_BITWISE_A,
        .left_index = SIZE_MAX,
        .right_index = SIZE_MAX,
    };
    chain->nodes[1] = (NNBitwiseChainNode){
        .kind = NN_BITWISE_CHAIN_INPUT_Y,
        .truth_mask = NN_BITWISE_B,
        .op_mask = NN_BITWISE_B,
        .left_index = SIZE_MAX,
        .right_index = SIZE_MAX,
    };
    chain->nodes[2] = (NNBitwiseChainNode){
        .kind = NN_BITWISE_CHAIN_CONST_ZERO,
        .truth_mask = NN_BITWISE_FALSE,
        .op_mask = NN_BITWISE_FALSE,
        .left_index = SIZE_MAX,
        .right_index = SIZE_MAX,
    };
    chain->nodes[3] = (NNBitwiseChainNode){
        .kind = NN_BITWISE_CHAIN_CONST_ONE,
        .truth_mask = NN_BITWISE_TRUE,
        .op_mask = NN_BITWISE_TRUE,
        .left_index = SIZE_MAX,
        .right_index = SIZE_MAX,
    };
    chain->node_count = BG_CHAIN_INITIAL_NODES;
    return 1;
}

static int bg_chain_find_mask(const NNBitwiseChain *chain, uint8_t truth_mask, size_t *index_out)
{
    if (chain == NULL) {
        return 0;
    }
    for (size_t i = 0; i < chain->node_count; ++i) {
        if (chain->nodes[i].truth_mask == (truth_mask & 0x0fu)) {
            if (index_out != NULL) {
                *index_out = i;
            }
            return 1;
        }
    }
    return 0;
}

static int bg_chain_append(NNBitwiseChain *chain,
                           uint8_t op_mask,
                           size_t left_index,
                           size_t right_index,
                           uint8_t result_mask)
{
    if (chain == NULL || chain->node_count >= NN_BITWISE_CHAIN_MAX_NODES ||
        left_index >= chain->node_count || right_index >= chain->node_count) {
        return 0;
    }
    chain->nodes[chain->node_count++] = (NNBitwiseChainNode){
        .kind = NN_BITWISE_CHAIN_DERIVED,
        .truth_mask = (uint8_t)(result_mask & 0x0fu),
        .op_mask = (uint8_t)(op_mask & 0x0fu),
        .left_index = left_index,
        .right_index = right_index,
    };
    chain->step_count = chain->node_count - BG_CHAIN_INITIAL_NODES;
    return 1;
}

static void bg_tool_features(nn_real *features,
                             uint8_t target_mask,
                             uint8_t left_mask,
                             uint8_t right_mask,
                             uint8_t op_mask,
                             uint8_t result_mask,
                             size_t node_count,
                             size_t step_index)
{
    memset(features, 0, NN_BITWISE_TOOL_FEATURE_COUNT * sizeof(*features));
    for (size_t bit = 0; bit < 4u; ++bit) {
        features[bit] = (nn_real)((target_mask >> bit) & 1u);
        features[4u + bit] = (nn_real)((left_mask >> bit) & 1u);
        features[8u + bit] = (nn_real)((right_mask >> bit) & 1u);
        features[12u + bit] = (nn_real)((op_mask >> bit) & 1u);
        features[16u + bit] = (nn_real)((result_mask >> bit) & 1u);
    }

    size_t matches = 0;
    for (size_t bit = 0; bit < 4u; ++bit) {
        matches += ((target_mask >> bit) & 1u) == ((result_mask >> bit) & 1u) ? 1u : 0u;
    }
    features[20] = (nn_real)((double)matches / 4.0);
    features[21] = result_mask == target_mask ? (nn_real)1 : (nn_real)0;
    features[22] = (nn_real)((double)node_count / (double)NN_BITWISE_CHAIN_MAX_NODES);
    features[23] = (nn_real)((double)step_index / (double)NN_BITWISE_CHAIN_MAX_NODES);
    features[24] = op_mask == NN_BITWISE_XOR ? (nn_real)1 : (nn_real)0;
    features[25] = op_mask == NN_BITWISE_AND ? (nn_real)1 : (nn_real)0;
    features[26] = op_mask == NN_BITWISE_OR ? (nn_real)1 : (nn_real)0;
    features[27] = op_mask == NN_BITWISE_NOT_A ? (nn_real)1 : (nn_real)0;
    features[28] = (nn_real)((double)(4u - matches) / 4.0);
    features[29] = left_mask == target_mask ? (nn_real)1 : (nn_real)0;
    features[30] = right_mask == target_mask ? (nn_real)1 : (nn_real)0;
    features[31] = (nn_real)1;
}

static size_t bg_count_allowed_ops(uint16_t allowed_truth_masks)
{
    size_t count = 0;
    for (uint8_t op = 0; op < NN_BITWISE_BINARY_TRUTH_TABLES; ++op) {
        count += bg_tool_is_allowed(allowed_truth_masks, op) ? 1u : 0u;
    }
    return count;
}

static size_t bg_enumerate_candidate_features(
    const NNBitwiseChain *chain,
    uint16_t allowed_truth_masks,
    uint8_t target_mask,
    nn_real *features,
    uint8_t *candidate_ops,
    size_t *candidate_left,
    size_t *candidate_right,
    uint8_t *candidate_results,
    size_t capacity)
{
    if (chain == NULL || features == NULL || candidate_ops == NULL ||
        candidate_left == NULL || candidate_right == NULL || candidate_results == NULL) {
        return 0;
    }

    size_t count = 0;
    for (uint8_t op = 0; op < NN_BITWISE_BINARY_TRUTH_TABLES; ++op) {
        if (!bg_tool_is_allowed(allowed_truth_masks, op)) {
            continue;
        }
        for (size_t left = 0; left < chain->node_count; ++left) {
            for (size_t right = 0; right < chain->node_count; ++right) {
                uint8_t result = nn_bitwise_apply_truth_mask_to_masks(
                    op,
                    chain->nodes[left].truth_mask,
                    chain->nodes[right].truth_mask);
                if (bg_chain_find_mask(chain, result, NULL)) {
                    continue;
                }
                if (count >= capacity) {
                    return count;
                }
                candidate_ops[count] = op;
                candidate_left[count] = left;
                candidate_right[count] = right;
                candidate_results[count] = result;
                bg_tool_features(&features[count * NN_BITWISE_TOOL_FEATURE_COUNT],
                                 target_mask,
                                 chain->nodes[left].truth_mask,
                                 chain->nodes[right].truth_mask,
                                 op,
                                 result,
                                 chain->node_count,
                                 chain->step_count);
                ++count;
            }
        }
    }
    return count;
}

typedef struct BGExpr {
    uint8_t truth_mask;
    uint8_t op_mask;
    int initial_node;
    struct BGExpr *left;
    struct BGExpr *right;
} BGExpr;

static int bg_initial_node_for_mask(uint8_t truth_mask, int *node_out)
{
    if (truth_mask == NN_BITWISE_A) {
        *node_out = 0;
        return 1;
    }
    if (truth_mask == NN_BITWISE_B) {
        *node_out = 1;
        return 1;
    }
    if (truth_mask == NN_BITWISE_FALSE) {
        *node_out = 2;
        return 1;
    }
    if (truth_mask == NN_BITWISE_TRUE) {
        *node_out = 3;
        return 1;
    }
    return 0;
}

static void bg_expr_free(BGExpr *expr)
{
    if (expr == NULL) {
        return;
    }
    bg_expr_free(expr->left);
    bg_expr_free(expr->right);
    free(expr);
}

static BGExpr *bg_expr_new(uint8_t truth_mask)
{
    BGExpr *expr = (BGExpr *)calloc(1u, sizeof(*expr));
    if (expr == NULL) {
        return NULL;
    }
    expr->truth_mask = (uint8_t)(truth_mask & 0x0fu);
    expr->initial_node = -1;
    return expr;
}

static BGExpr *bg_expr_find(uint16_t allowed_truth_masks, uint8_t target_truth_mask, size_t depth)
{
    int initial_node = -1;
    target_truth_mask = (uint8_t)(target_truth_mask & 0x0fu);
    if (bg_initial_node_for_mask(target_truth_mask, &initial_node)) {
        BGExpr *expr = bg_expr_new(target_truth_mask);
        if (expr != NULL) {
            expr->initial_node = initial_node;
        }
        return expr;
    }
    if (depth == 0u) {
        return NULL;
    }

    for (uint8_t op = 0; op < NN_BITWISE_BINARY_TRUTH_TABLES; ++op) {
        if (!bg_tool_is_allowed(allowed_truth_masks, op)) {
            continue;
        }
        for (uint8_t left = 0; left < NN_BITWISE_BINARY_TRUTH_TABLES; ++left) {
            for (uint8_t right = 0; right < NN_BITWISE_BINARY_TRUTH_TABLES; ++right) {
                uint8_t result = nn_bitwise_apply_truth_mask_to_masks(op, left, right);
                if (result != target_truth_mask) {
                    continue;
                }
                BGExpr *left_expr = bg_expr_find(allowed_truth_masks, left, depth - 1u);
                if (left_expr == NULL) {
                    continue;
                }
                BGExpr *right_expr = bg_expr_find(allowed_truth_masks, right, depth - 1u);
                if (right_expr == NULL) {
                    bg_expr_free(left_expr);
                    continue;
                }
                BGExpr *expr = bg_expr_new(target_truth_mask);
                if (expr == NULL) {
                    bg_expr_free(left_expr);
                    bg_expr_free(right_expr);
                    return NULL;
                }
                expr->op_mask = op;
                expr->left = left_expr;
                expr->right = right_expr;
                return expr;
            }
        }
    }
    return NULL;
}

static int bg_chain_append_expr(NNBitwiseChain *chain, const BGExpr *expr, size_t *index_out)
{
    if (chain == NULL || expr == NULL || index_out == NULL) {
        return 0;
    }
    size_t existing = 0;
    if (bg_chain_find_mask(chain, expr->truth_mask, &existing)) {
        *index_out = existing;
        return 1;
    }
    if (expr->initial_node >= 0) {
        *index_out = (size_t)expr->initial_node;
        return 1;
    }

    size_t left_index = 0;
    size_t right_index = 0;
    if (!bg_chain_append_expr(chain, expr->left, &left_index) ||
        !bg_chain_append_expr(chain, expr->right, &right_index) ||
        !bg_chain_append(chain,
                         expr->op_mask,
                         left_index,
                         right_index,
                         expr->truth_mask)) {
        return 0;
    }
    *index_out = chain->node_count - 1u;
    return 1;
}

static int bg_chain_find_shortest(uint16_t allowed_truth_masks,
                                  size_t max_steps,
                                  uint8_t target_truth_mask,
                                  NNBitwiseChain *out_chain)
{
    if (out_chain == NULL || bg_count_allowed_ops(allowed_truth_masks) == 0u ||
        !bg_chain_seed(out_chain)) {
        return 0;
    }

    for (size_t depth = 0; depth <= max_steps; ++depth) {
        BGExpr *expr = bg_expr_find(allowed_truth_masks, target_truth_mask, depth);
        if (expr == NULL) {
            continue;
        }
        NNBitwiseChain chain;
        size_t output_index = SIZE_MAX;
        int ok = bg_chain_seed(&chain) && bg_chain_append_expr(&chain, expr, &output_index);
        bg_expr_free(expr);
        if (ok) {
            chain.output_index = output_index;
            chain.target_truth_mask = (uint8_t)(target_truth_mask & 0x0fu);
            chain.solved = 1;
            *out_chain = chain;
            return 1;
        }
    }
    return 0;
}

int nn_bitwise_tool_policy_train(NNBitwiseToolPolicy *policy,
                                 uint16_t allowed_truth_masks,
                                 size_t max_steps,
                                 const NNBitwiseToolPolicyConfig *config)
{
    if (policy == NULL || allowed_truth_masks == 0u || max_steps == 0u) {
        return 0;
    }

    NNBitwiseToolPolicyConfig default_config;
    NNBitwiseToolPolicyConfig active_config;
    if (config == NULL) {
        nn_bitwise_tool_policy_config_init(&default_config);
        active_config = default_config;
    } else {
        active_config = *config;
    }
    if (active_config.hidden_width == 0u) {
        active_config.hidden_width = 1u;
    }
    if (active_config.epochs == 0u) {
        active_config.epochs = 1u;
    }
    if (active_config.learning_rate <= (nn_real)0 || !bg_finite(active_config.learning_rate)) {
        active_config.learning_rate = (nn_real)0.05;
    }

    nn_bitwise_tool_policy_free(policy);
    policy->allowed_truth_masks = allowed_truth_masks;
    policy->max_steps = max_steps;
    policy->final_loss = (nn_real)NAN;

    size_t layers[] = {NN_BITWISE_TOOL_FEATURE_COUNT, active_config.hidden_width, 1u};
    NNConfig nn_config = {
        .layer_sizes = layers,
        .layer_count = sizeof(layers) / sizeof(layers[0]),
        .seed = active_config.seed,
        .hidden_activation = NN_ACTIVATION_TANH,
        .output_activation = NN_ACTIVATION_LINEAR,
        .loss = NN_LOSS_MSE,
    };
    policy->net = nn_create_config(&nn_config);
    if (policy->net == NULL) {
        return 0;
    }

    size_t allowed_count = bg_count_allowed_ops(allowed_truth_masks);
    size_t max_candidates =
        allowed_count * NN_BITWISE_CHAIN_MAX_NODES * NN_BITWISE_CHAIN_MAX_NODES;
    nn_real *candidate_features =
        (nn_real *)calloc(max_candidates * NN_BITWISE_TOOL_FEATURE_COUNT,
                          sizeof(*candidate_features));
    uint8_t *candidate_ops = (uint8_t *)calloc(max_candidates, sizeof(*candidate_ops));
    size_t *candidate_left = (size_t *)calloc(max_candidates, sizeof(*candidate_left));
    size_t *candidate_right = (size_t *)calloc(max_candidates, sizeof(*candidate_right));
    uint8_t *candidate_results =
        (uint8_t *)calloc(max_candidates, sizeof(*candidate_results));
    if (candidate_features == NULL || candidate_ops == NULL || candidate_left == NULL ||
        candidate_right == NULL || candidate_results == NULL) {
        free(candidate_features);
        free(candidate_ops);
        free(candidate_left);
        free(candidate_right);
        free(candidate_results);
        return 0;
    }

    NNBitwiseChain oracles[NN_BITWISE_BINARY_TRUTH_TABLES];
    size_t solved = 0;
    size_t groups = 0;
    for (uint8_t target = 0; target < NN_BITWISE_BINARY_TRUTH_TABLES; ++target) {
        nn_bitwise_chain_init(&oracles[target]);
        if (bg_chain_find_shortest(allowed_truth_masks, max_steps, target, &oracles[target])) {
            ++solved;
            groups += oracles[target].step_count;
        }
    }
    policy->training_groups = groups;
    policy->oracle_coverage = (double)solved / (double)NN_BITWISE_BINARY_TRUTH_TABLES;

    nn_real loss = (nn_real)0;
    for (size_t epoch = 0; epoch < active_config.epochs; ++epoch) {
        loss = (nn_real)0;
        for (uint8_t target = 0; target < NN_BITWISE_BINARY_TRUTH_TABLES; ++target) {
            const NNBitwiseChain *oracle = &oracles[target];
            if (!oracle->solved || oracle->step_count == 0u) {
                continue;
            }
            NNBitwiseChain prefix;
            bg_chain_seed(&prefix);
            for (size_t step = BG_CHAIN_INITIAL_NODES; step < oracle->node_count; ++step) {
                const NNBitwiseChainNode *wanted = &oracle->nodes[step];
                size_t candidate_count = bg_enumerate_candidate_features(&prefix,
                                                                         allowed_truth_masks,
                                                                         target,
                                                                         candidate_features,
                                                                         candidate_ops,
                                                                         candidate_left,
                                                                         candidate_right,
                                                                         candidate_results,
                                                                         max_candidates);
                size_t target_index = SIZE_MAX;
                for (size_t i = 0; i < candidate_count; ++i) {
                    if (candidate_ops[i] == wanted->op_mask &&
                        candidate_left[i] == wanted->left_index &&
                        candidate_right[i] == wanted->right_index &&
                        candidate_results[i] == wanted->truth_mask) {
                        target_index = i;
                        break;
                    }
                }
                if (target_index != SIZE_MAX && candidate_count > 0u) {
                    loss += nn_train_softmax_scorer(policy->net,
                                                    candidate_features,
                                                    candidate_count,
                                                    target_index,
                                                    active_config.learning_rate);
                    ++policy->training_updates;
                }
                bg_chain_append(&prefix,
                                wanted->op_mask,
                                wanted->left_index,
                                wanted->right_index,
                                wanted->truth_mask);
            }
        }
    }
    policy->final_loss =
        policy->training_updates == 0u ? (nn_real)NAN : loss / (nn_real)(groups == 0u ? 1u : groups);

    free(candidate_features);
    free(candidate_ops);
    free(candidate_left);
    free(candidate_right);
    free(candidate_results);
    return solved == NN_BITWISE_BINARY_TRUTH_TABLES;
}

int nn_bitwise_chain_solve_truth_table(const NNBitwiseToolPolicy *policy,
                                       const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
                                       uint8_t target_truth_mask,
                                       NNBitwiseChain *chain)
{
    if (policy == NULL || policy->net == NULL || chain == NULL ||
        (target_truth_mask & 0xf0u) != 0u) {
        return 0;
    }
    for (uint8_t op = 0; op < NN_BITWISE_BINARY_TRUTH_TABLES; ++op) {
        if (bg_tool_is_allowed(policy->allowed_truth_masks, op) &&
            (gates == NULL || gates[op].certificate.status != NN_CERTIFICATE_CERTIFIED)) {
            return 0;
        }
    }

    size_t allowed_count = bg_count_allowed_ops(policy->allowed_truth_masks);
    size_t max_candidates =
        allowed_count * NN_BITWISE_CHAIN_MAX_NODES * NN_BITWISE_CHAIN_MAX_NODES;
    nn_real *candidate_features =
        (nn_real *)calloc(max_candidates * NN_BITWISE_TOOL_FEATURE_COUNT,
                          sizeof(*candidate_features));
    nn_real *scores = (nn_real *)calloc(max_candidates, sizeof(*scores));
    nn_real *probabilities = (nn_real *)calloc(max_candidates, sizeof(*probabilities));
    size_t *ranked = (size_t *)calloc(max_candidates, sizeof(*ranked));
    uint8_t *candidate_ops = (uint8_t *)calloc(max_candidates, sizeof(*candidate_ops));
    size_t *candidate_left = (size_t *)calloc(max_candidates, sizeof(*candidate_left));
    size_t *candidate_right = (size_t *)calloc(max_candidates, sizeof(*candidate_right));
    uint8_t *candidate_results =
        (uint8_t *)calloc(max_candidates, sizeof(*candidate_results));
    if (candidate_features == NULL || scores == NULL || probabilities == NULL ||
        ranked == NULL || candidate_ops == NULL || candidate_left == NULL ||
        candidate_right == NULL || candidate_results == NULL || !bg_chain_seed(chain)) {
        free(candidate_features);
        free(scores);
        free(probabilities);
        free(ranked);
        free(candidate_ops);
        free(candidate_left);
        free(candidate_right);
        free(candidate_results);
        return 0;
    }
    chain->target_truth_mask = (uint8_t)(target_truth_mask & 0x0fu);

    size_t index = 0;
    if (bg_chain_find_mask(chain, target_truth_mask, &index)) {
        chain->output_index = index;
        chain->solved = 1;
        goto done;
    }

    for (size_t step = 0; step < policy->max_steps; ++step) {
        size_t candidate_count = bg_enumerate_candidate_features(chain,
                                                                 policy->allowed_truth_masks,
                                                                 target_truth_mask,
                                                                 candidate_features,
                                                                 candidate_ops,
                                                                 candidate_left,
                                                                 candidate_right,
                                                                 candidate_results,
                                                                 max_candidates);
        if (candidate_count == 0u ||
            !nn_score_action_rows(policy->net,
                                  candidate_features,
                                  candidate_count,
                                  scores,
                                  probabilities) ||
            !nn_rank_action_scores(scores, candidate_count, ranked)) {
            break;
        }

        int appended = 0;
        for (size_t r = 0; r < candidate_count; ++r) {
            size_t i = ranked[r];
            if (bg_chain_append(chain,
                                candidate_ops[i],
                                candidate_left[i],
                                candidate_right[i],
                                candidate_results[i])) {
                appended = 1;
                break;
            }
        }
        if (!appended) {
            break;
        }
        if (bg_chain_find_mask(chain, target_truth_mask, &index)) {
            chain->output_index = index;
            chain->solved = 1;
            goto done;
        }
    }

    if (bg_chain_find_shortest(policy->allowed_truth_masks,
                               policy->max_steps,
                               target_truth_mask,
                               chain)) {
        chain->used_symbolic_fallback = 1;
        goto done;
    }

done:
    free(candidate_features);
    free(scores);
    free(probabilities);
    free(ranked);
    free(candidate_ops);
    free(candidate_left);
    free(candidate_right);
    free(candidate_results);
    return chain->solved;
}

int nn_bitwise_infer_grid_truth_constraints(const NNBitGrid *x,
                                            const NNBitGrid *y,
                                            const NNBitGrid *target,
                                            uint8_t *required_truth_mask,
                                            uint8_t *observed_rows_mask)
{
    if (!nn_bitgrid_same_shape(x, y) || !nn_bitgrid_same_shape(x, target) ||
        required_truth_mask == NULL || observed_rows_mask == NULL) {
        return 0;
    }

    uint8_t required = 0u;
    uint8_t observed = 0u;
    size_t cells = nn_bitgrid_cell_count(x);
    for (size_t i = 0; i < cells; ++i) {
        uint8_t row = (uint8_t)(((x->cells[i] & 1u) << 1u) | (y->cells[i] & 1u));
        uint8_t bit = (uint8_t)(target->cells[i] & 1u);
        uint8_t row_mask = (uint8_t)(1u << row);
        if ((observed & row_mask) != 0u) {
            uint8_t previous = (uint8_t)((required >> row) & 1u);
            if (previous != bit) {
                return 0;
            }
        }
        observed |= row_mask;
        if (bit != 0u) {
            required |= row_mask;
        } else {
            required &= (uint8_t)~row_mask;
        }
    }

    *required_truth_mask = required;
    *observed_rows_mask = observed;
    return 1;
}

static int bg_truth_mask_matches_constraints(uint8_t truth_mask,
                                             uint8_t required_truth_mask,
                                             uint8_t observed_rows_mask)
{
    return (((truth_mask ^ required_truth_mask) & observed_rows_mask) == 0u);
}

int nn_bitwise_chain_apply(const NNBitwiseChain *chain,
                           const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
                           const NNBitGrid *x,
                           const NNBitGrid *y,
                           NNBitGrid *out)
{
    if (chain == NULL || !chain->solved || chain->node_count == 0u ||
        chain->node_count > NN_BITWISE_CHAIN_MAX_NODES ||
        chain->output_index >= chain->node_count || gates == NULL ||
        !nn_bitgrid_same_shape(x, y) || !nn_bitgrid_same_shape(x, out)) {
        return 0;
    }

    NNBitGrid values[NN_BITWISE_CHAIN_MAX_NODES];
    memset(values, 0, sizeof(values));
    int ok = 1;
    for (size_t i = 0; ok && i < chain->node_count; ++i) {
        ok = nn_bitgrid_init(&values[i], x->width, x->height);
        if (!ok) {
            break;
        }
        const NNBitwiseChainNode *node = &chain->nodes[i];
        switch (node->kind) {
        case NN_BITWISE_CHAIN_INPUT_X:
            ok = nn_bitgrid_copy(&values[i], x);
            break;
        case NN_BITWISE_CHAIN_INPUT_Y:
            ok = nn_bitgrid_copy(&values[i], y);
            break;
        case NN_BITWISE_CHAIN_CONST_ZERO:
            ok = nn_bitgrid_fill(&values[i], 0u);
            break;
        case NN_BITWISE_CHAIN_CONST_ONE:
            ok = nn_bitgrid_fill(&values[i], 1u);
            break;
        case NN_BITWISE_CHAIN_DERIVED:
            if (node->op_mask >= NN_BITWISE_BINARY_TRUTH_TABLES ||
                node->left_index >= i || node->right_index >= i ||
                gates[node->op_mask].certificate.status != NN_CERTIFICATE_CERTIFIED) {
                ok = 0;
            } else {
                ok = nn_bitwise_gate_apply(&gates[node->op_mask],
                                           &values[node->left_index],
                                           &values[node->right_index],
                                           &values[i]);
            }
            break;
        default:
            ok = 0;
            break;
        }
    }

    if (ok) {
        ok = nn_bitgrid_copy(out, &values[chain->output_index]);
    }

    for (size_t i = 0; i < chain->node_count; ++i) {
        nn_bitgrid_free(&values[i]);
    }
    return ok;
}

int nn_bitwise_chain_eval_truth_mask(
    const NNBitwiseChain *chain,
    const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
    uint8_t *out_truth_mask)
{
    if (chain == NULL || !chain->solved || chain->node_count == 0u ||
        chain->node_count > NN_BITWISE_CHAIN_MAX_NODES ||
        chain->output_index >= chain->node_count || gates == NULL ||
        out_truth_mask == NULL) {
        return 0;
    }

    uint8_t values[NN_BITWISE_CHAIN_MAX_NODES];
    memset(values, 0, sizeof(values));
    for (size_t i = 0; i < chain->node_count; ++i) {
        const NNBitwiseChainNode *node = &chain->nodes[i];
        uint8_t value = 0u;
        switch (node->kind) {
        case NN_BITWISE_CHAIN_INPUT_X:
            value = NN_BITWISE_A;
            break;
        case NN_BITWISE_CHAIN_INPUT_Y:
            value = NN_BITWISE_B;
            break;
        case NN_BITWISE_CHAIN_CONST_ZERO:
            value = NN_BITWISE_FALSE;
            break;
        case NN_BITWISE_CHAIN_CONST_ONE:
            value = NN_BITWISE_TRUE;
            break;
        case NN_BITWISE_CHAIN_DERIVED:
            if (node->op_mask >= NN_BITWISE_BINARY_TRUTH_TABLES ||
                node->left_index >= i || node->right_index >= i ||
                gates[node->op_mask].certificate.status != NN_CERTIFICATE_CERTIFIED) {
                return 0;
            }
            value = nn_bitwise_apply_truth_mask_to_masks(node->op_mask,
                                                         values[node->left_index],
                                                         values[node->right_index]);
            break;
        default:
            return 0;
        }
        if (value != (node->truth_mask & 0x0fu)) {
            return 0;
        }
        values[i] = value;
    }

    *out_truth_mask = values[chain->output_index];
    return 1;
}

int nn_bitwise_chain_solve_grid_transform(
    const NNBitwiseToolPolicy *policy,
    const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
    const NNBitGrid *x,
    const NNBitGrid *y,
    const NNBitGrid *target,
    NNBitwiseChain *chain)
{
    if (policy == NULL || gates == NULL || chain == NULL || !nn_bitgrid_same_shape(x, y) ||
        !nn_bitgrid_same_shape(x, target)) {
        return 0;
    }

    uint8_t required = 0u;
    uint8_t observed = 0u;
    if (!nn_bitwise_infer_grid_truth_constraints(x, y, target, &required, &observed)) {
        return 0;
    }

    NNBitGrid candidate = {0};
    if (!nn_bitgrid_init(&candidate, target->width, target->height)) {
        return 0;
    }

    int ok = 0;
    for (uint8_t truth_mask = 0u; truth_mask < NN_BITWISE_BINARY_TRUTH_TABLES; ++truth_mask) {
        if (!bg_truth_mask_matches_constraints(truth_mask, required, observed)) {
            continue;
        }
        NNBitwiseChain attempt;
        nn_bitwise_chain_init(&attempt);
        if (!nn_bitwise_chain_solve_truth_table(policy, gates, truth_mask, &attempt)) {
            continue;
        }
        if (nn_bitwise_chain_apply(&attempt, gates, x, y, &candidate) &&
            nn_bitgrid_hamming_distance(&candidate, target) == 0u) {
            *chain = attempt;
            ok = 1;
            break;
        }
    }

    nn_bitgrid_free(&candidate);
    return ok;
}

int nn_bitwise_chain_write(FILE *fp, const NNBitwiseChain *chain)
{
    if (fp == NULL || chain == NULL) {
        return 0;
    }
    fprintf(fp,
            "target=%s solved=%d steps=%zu output=%zu fallback=%d\n",
            nn_bitwise_truth_mask_name(chain->target_truth_mask),
            chain->solved,
            chain->step_count,
            chain->output_index,
            chain->used_symbolic_fallback);
    for (size_t i = 0; i < chain->node_count; ++i) {
        const NNBitwiseChainNode *node = &chain->nodes[i];
        if (node->kind == NN_BITWISE_CHAIN_DERIVED) {
            fprintf(fp,
                    "  n%zu=%s(n%zu:%s,n%zu:%s)->%s\n",
                    i,
                    nn_bitwise_truth_mask_name(node->op_mask),
                    node->left_index,
                    nn_bitwise_truth_mask_name(chain->nodes[node->left_index].truth_mask),
                    node->right_index,
                    nn_bitwise_truth_mask_name(chain->nodes[node->right_index].truth_mask),
                    nn_bitwise_truth_mask_name(node->truth_mask));
        } else {
            const char *kind = node->kind == NN_BITWISE_CHAIN_INPUT_X
                                   ? "X"
                                   : node->kind == NN_BITWISE_CHAIN_INPUT_Y
                                         ? "Y"
                                         : node->kind == NN_BITWISE_CHAIN_CONST_ZERO ? "0" : "1";
            fprintf(fp, "  n%zu=%s:%s\n", i, kind, nn_bitwise_truth_mask_name(node->truth_mask));
        }
    }
    return ferror(fp) == 0;
}

void nn_bitwise_macro_library_init(NNBitwiseMacroLibrary *library)
{
    if (library == NULL) {
        return;
    }
    memset(library, 0, sizeof(*library));
    for (size_t i = 0; i < NN_BITWISE_MACRO_MAX_TOOLS; ++i) {
        snprintf(library->tools[i].schema,
                 sizeof(library->tools[i].schema),
                 "%s",
                 NN_BITWISE_MACRO_SCHEMA);
        nn_certificate_clear(&library->tools[i].certificate);
        nn_bitwise_chain_init(&library->tools[i].chain);
    }
}

int nn_bitwise_macro_library_learn(
    NNBitwiseMacroLibrary *library,
    const NNBitwiseToolPolicy *policy,
    const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
    uint16_t target_truth_masks)
{
    if (library == NULL || policy == NULL || gates == NULL || target_truth_masks == 0u) {
        return 0;
    }

    nn_bitwise_macro_library_init(library);
    for (uint8_t target = 0; target < NN_BITWISE_BINARY_TRUTH_TABLES; ++target) {
        if (((target_truth_masks >> target) & 1u) == 0u) {
            continue;
        }
        if (library->tool_count >= NN_BITWISE_MACRO_MAX_TOOLS) {
            return 0;
        }
        NNBitwiseMacroTool *tool = &library->tools[library->tool_count++];
        snprintf(tool->schema, sizeof(tool->schema), "%s", NN_BITWISE_MACRO_SCHEMA);
        tool->truth_mask = target;
        tool->support = BG_TRUTH_ROWS;
        ++library->candidate_count;

        NNBitwiseChain chain;
        nn_bitwise_chain_init(&chain);
        uint8_t actual = 0u;
        if (!nn_bitwise_chain_solve_truth_table(policy, gates, target, &chain)) {
            nn_certificate_reject(&tool->certificate,
                                  "macro_chain_search",
                                  "no chain found within policy depth",
                                  BG_TRUTH_ROWS,
                                  BG_TRUTH_ROWS);
            continue;
        }
        tool->chain = chain;
        tool->step_count = chain.step_count;
        tool->used_symbolic_fallback = chain.used_symbolic_fallback;
        if (!nn_bitwise_chain_eval_truth_mask(&tool->chain, gates, &actual) ||
            actual != target) {
            nn_certificate_reject(&tool->certificate,
                                  "macro_composition_replay",
                                  "chain replay did not match target truth mask",
                                  BG_TRUTH_ROWS,
                                  BG_TRUTH_ROWS);
            continue;
        }

        tool->promoted = 1;
        nn_certificate_accept(&tool->certificate,
                              "macro_composition_replay",
                              "chain of certified gates replays to target truth mask",
                              BG_TRUTH_ROWS,
                              BG_TRUTH_ROWS);
        library->promoted_truth_masks |= (uint16_t)(1u << target);
        ++library->promoted_count;
        library->fallback_promoted_count += tool->used_symbolic_fallback ? 1u : 0u;
    }

    return library->promoted_count == library->candidate_count;
}

const NNBitwiseMacroTool *nn_bitwise_macro_library_find(const NNBitwiseMacroLibrary *library,
                                                        uint8_t truth_mask)
{
    if (library == NULL || (truth_mask & 0xf0u) != 0u) {
        return NULL;
    }
    for (size_t i = 0; i < library->tool_count; ++i) {
        const NNBitwiseMacroTool *tool = &library->tools[i];
        if (tool->promoted && tool->truth_mask == truth_mask &&
            tool->certificate.status == NN_CERTIFICATE_CERTIFIED) {
            return tool;
        }
    }
    return NULL;
}

int nn_bitwise_macro_tool_apply(const NNBitwiseMacroTool *tool,
                                const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
                                const NNBitGrid *x,
                                const NNBitGrid *y,
                                NNBitGrid *out)
{
    if (tool == NULL || !tool->promoted ||
        tool->certificate.status != NN_CERTIFICATE_CERTIFIED) {
        return 0;
    }
    return nn_bitwise_chain_apply(&tool->chain, gates, x, y, out);
}

int nn_bitwise_macro_library_write_json(FILE *fp, const NNBitwiseMacroLibrary *library)
{
    if (fp == NULL || library == NULL) {
        return 0;
    }
    fprintf(fp,
            "{\"schema\":\"%s.summary\","
            "\"candidate_count\":%zu,"
            "\"promoted_count\":%zu,"
            "\"fallback_promoted_count\":%zu,"
            "\"promoted_truth_masks\":%u}\n",
            NN_BITWISE_MACRO_SCHEMA,
            library->candidate_count,
            library->promoted_count,
            library->fallback_promoted_count,
            (unsigned)library->promoted_truth_masks);
    for (size_t i = 0; i < library->tool_count; ++i) {
        const NNBitwiseMacroTool *tool = &library->tools[i];
        fprintf(fp,
                "{\"schema\":\"%s\","
                "\"op\":\"%s\","
                "\"truth_mask\":%u,"
                "\"steps\":%zu,"
                "\"support\":%zu,"
                "\"certificate\":\"%s\","
                "\"promoted\":%s,"
                "\"fallback\":%s}\n",
                NN_BITWISE_MACRO_SCHEMA,
                nn_bitwise_truth_mask_name(tool->truth_mask),
                (unsigned)tool->truth_mask,
                tool->step_count,
                tool->support,
                nn_certificate_status_name(tool->certificate.status),
                tool->promoted ? "true" : "false",
                tool->used_symbolic_fallback ? "true" : "false");
    }
    return ferror(fp) == 0;
}

int nn_bitwise_gate_train_truth_table(NNBitwiseGate *gate,
                                      uint8_t truth_mask,
                                      const NNBitwiseTrainConfig *config)
{
    if (gate == NULL || (truth_mask & 0xf0u) != 0u) {
        return 0;
    }

    NNBitwiseTrainConfig default_config;
    NNBitwiseTrainConfig active_config;
    if (config == NULL) {
        nn_bitwise_train_config_init(&default_config);
        active_config = default_config;
    } else {
        active_config = *config;
    }
    if (active_config.hidden_width == 0u) {
        active_config.hidden_width = 1u;
    }
    if (active_config.max_epochs == 0u) {
        active_config.max_epochs = 1u;
    }
    if (active_config.batch_size == 0u || active_config.batch_size > BG_TRUTH_ROWS) {
        active_config.batch_size = BG_TRUTH_ROWS;
    }
    if (active_config.retry_count == 0u) {
        active_config.retry_count = 1u;
    }
    if (active_config.learning_rate <= (nn_real)0 || !bg_finite(active_config.learning_rate)) {
        active_config.learning_rate = (nn_real)0.08;
    }
    if (!bg_finite(active_config.min_margin)) {
        active_config.min_margin = (nn_real)0;
    }

    nn_bitwise_gate_free(gate);
    gate->op = (NNBitwiseOp)(truth_mask & 0x0fu);
    gate->truth_mask = (uint8_t)(truth_mask & 0x0fu);
    gate->arity = 2u;
    bg_truth_table(gate->truth_mask, gate->truth_table);
    bg_report_init(&gate->report, gate->truth_mask);

    nn_real inputs[BG_TRUTH_ROWS * BG_INPUT_WIDTH];
    nn_real targets[BG_TRUTH_ROWS * BG_OUTPUT_WIDTH];
    bg_build_dataset(gate->truth_mask, inputs, targets);

    NNBitwiseTrainReport best_report;
    bg_report_init(&best_report, gate->truth_mask);
    uint8_t best_predicted[BG_TRUTH_ROWS] = {0, 0, 0, 0};

    for (size_t attempt = 0; attempt < active_config.retry_count; ++attempt) {
        size_t layers[] = {BG_INPUT_WIDTH, active_config.hidden_width, active_config.hidden_width, active_config.hidden_width, BG_OUTPUT_WIDTH};
        NNConfig nn_config = {
            .layer_sizes = layers,
            .layer_count = sizeof(layers) / sizeof(layers[0]),
            .seed = active_config.seed + (unsigned int)(attempt * 65537u),
            .hidden_activation = NN_ACTIVATION_TANH,
            .output_activation = NN_ACTIVATION_LINEAR,
            .loss = NN_LOSS_SOFTMAX_CROSS_ENTROPY,
        };
        NeuralNet *net = nn_create_config(&nn_config);
        if (net == NULL) {
            continue;
        }

        NNBitwiseTrainReport attempt_report;
        uint8_t predicted[BG_TRUTH_ROWS] = {0, 0, 0, 0};
        bg_report_init(&attempt_report, gate->truth_mask);
        attempt_report.attempts = attempt + 1u;

        for (size_t epochs = 0; epochs < active_config.max_epochs;) {
            size_t chunk = bg_min_size(BG_TRAIN_CHUNK_EPOCHS, active_config.max_epochs - epochs);
            nn_real loss = nn_train_batched(net,
                                            inputs,
                                            targets,
                                            BG_TRUTH_ROWS,
                                            active_config.batch_size,
                                            active_config.learning_rate,
                                            chunk,
                                            active_config.shuffle);
            epochs += chunk;
            attempt_report.epochs = epochs;
            attempt_report.final_loss = loss;
            bg_eval_net(net, gate->truth_mask, predicted, &attempt_report);
            if (bg_report_is_better(&attempt_report, &best_report)) {
                best_report = attempt_report;
                memcpy(best_predicted, predicted, sizeof(best_predicted));
            }
            if (attempt_report.learned_truth_table &&
                attempt_report.min_margin >= active_config.min_margin) {
                gate->net = net;
                memcpy(gate->predicted_table, predicted, sizeof(gate->predicted_table));
                gate->report = attempt_report;
                gate->report.promoted = 1;
                nn_certificate_accept(&gate->certificate,
                                      "exhaustive_truth_table",
                                      "neural truth table matches every binary input row",
                                      BG_TRUTH_ROWS,
                                      BG_TRUTH_ROWS);
                return 1;
            }
        }

        nn_free(net);
    }

    gate->report = best_report;
    memcpy(gate->predicted_table, best_predicted, sizeof(gate->predicted_table));
    size_t correct = (size_t)(best_report.accuracy * (double)BG_TRUTH_ROWS + 0.5);
    nn_certificate_reject(&gate->certificate,
                          "exhaustive_truth_table",
                          "neural truth table did not meet promotion margin",
                          BG_TRUTH_ROWS,
                          BG_TRUTH_ROWS - correct);
    return 0;
}

int nn_bitwise_gate_train(NNBitwiseGate *gate,
                          NNBitwiseOp op,
                          const NNBitwiseTrainConfig *config)
{
    return nn_bitwise_gate_train_truth_table(gate, nn_bitwise_op_truth_mask(op), config);
}

int nn_bitwise_gate_predict_cell(const NNBitwiseGate *gate,
                                 uint8_t a,
                                 uint8_t b,
                                 uint8_t *out_bit,
                                 nn_real *confidence)
{
    if (gate == NULL || out_bit == NULL) {
        return 0;
    }

    if (gate->certificate.status == NN_CERTIFICATE_CERTIFIED) {
        if (!nn_bitwise_truth_mask_eval(gate->truth_mask, a, b, out_bit)) {
            return 0;
        }
        if (confidence != NULL) {
            *confidence = (nn_real)1;
        }
        return 1;
    }

    if (gate->net == NULL) {
        return 0;
    }

    nn_real input[BG_INPUT_WIDTH] = {(nn_real)(a & 1u), (nn_real)(b & 1u)};
    nn_real logits[BG_OUTPUT_WIDTH];
    nn_real probabilities[BG_OUTPUT_WIDTH];
    nn_forward(gate->net, input, logits);
    if (!bg_finite(logits[0]) || !bg_finite(logits[1])) {
        return 0;
    }
    nn_softmax(logits, probabilities, BG_OUTPUT_WIDTH);
    if (!bg_finite(probabilities[0]) || !bg_finite(probabilities[1])) {
        return 0;
    }
    *out_bit = probabilities[1] >= probabilities[0] ? 1u : 0u;
    if (confidence != NULL) {
        *confidence = probabilities[*out_bit];
    }
    return 1;
}

int nn_bitwise_gate_apply(const NNBitwiseGate *gate,
                          const NNBitGrid *a,
                          const NNBitGrid *b,
                          NNBitGrid *out)
{
    if (gate == NULL || !nn_bitgrid_same_shape(a, b) || !nn_bitgrid_same_shape(a, out)) {
        return 0;
    }

    size_t cells = nn_bitgrid_cell_count(a);
    for (size_t i = 0; i < cells; ++i) {
        uint8_t result = 0;
        if (!nn_bitwise_gate_predict_cell(gate, a->cells[i], b->cells[i], &result, NULL)) {
            return 0;
        }
        out->cells[i] = result;
    }
    return 1;
}

int nn_bitwise_gate_write_json(FILE *fp, const NNBitwiseGate *gate)
{
    if (fp == NULL || gate == NULL) {
        return 0;
    }

    fprintf(fp,
            "{\"schema\":\"%s\","
            "\"op\":\"%s\","
            "\"truth_mask\":%u,"
            "\"arity\":%zu,"
            "\"parameters\":%zu,"
            "\"epochs\":%zu,"
            "\"attempts\":%zu,"
            "\"accuracy\":%.8f,"
            "\"min_margin\":%.8f,"
            "\"final_loss\":%.8f,"
            "\"certificate\":\"%s\","
            "\"promoted\":%s}\n",
            NN_BITGRID_GATE_SCHEMA,
            nn_bitwise_truth_mask_name(gate->truth_mask),
            (unsigned)gate->truth_mask,
            gate->arity,
            gate->net == NULL ? 0u : nn_parameter_count(gate->net),
            gate->report.epochs,
            gate->report.attempts,
            gate->report.accuracy,
            (double)gate->report.min_margin,
            (double)gate->report.final_loss,
            nn_certificate_status_name(gate->certificate.status),
            gate->report.promoted ? "true" : "false");
    return ferror(fp) == 0;
}
